"""Fixed-horizon synthetic fluid benchmark, with no field-performance claim.

Raw readings, their covariance and fault templates all pass through the same
measurement operator. Intervention labels are used only in scoring. The nominal
flow profiles and onset are declared design information, not estimated from test
readings. Consequently this is a controlled identifiability experiment, not an
unknown-onset operational diagnostic or an online detection-delay benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np

from set_lcm.diagnostics import diagnose
from set_lcm.measurement import BalanceRecord, balance_residuals

from .provenance import header_line, provenance


@dataclass(frozen=True)
class ExperimentConfig:
    intervals: int = 24
    interval_seconds: float = 1.0
    onset_interval: int = 6
    alpha: float = 0.01
    interval_level: float = 0.95
    storage_sd_m3: float = 0.12
    flow_sd_m3_per_s: float = 0.025
    reference_storage_m3: float = 0.08
    reference_inflow_m3_per_s: float = 0.015
    reference_outflow_m3_per_s: float = 0.010
    weak_storage_m3: float = 0.02
    strong_storage_m3: float = 1.6
    weak_rate_m3_per_s: float = 0.001
    strong_rate_m3_per_s: float = 0.16
    weak_gain: float = 0.0005
    strong_gain: float = 0.08

    def __post_init__(self) -> None:
        for name in ("intervals", "onset_interval"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, not a boolean")
            object.__setattr__(self, name, int(value))
        if self.intervals < 8 or not 1 <= self.onset_interval < self.intervals - 3:
            raise ValueError("need at least eight intervals and an interior known onset")
        positive = ("interval_seconds", "storage_sd_m3", "flow_sd_m3_per_s")
        references = ("reference_storage_m3", "reference_inflow_m3_per_s", "reference_outflow_m3_per_s")
        amplitudes = ("weak_storage_m3", "strong_storage_m3", "weak_rate_m3_per_s",
                      "strong_rate_m3_per_s", "weak_gain", "strong_gain")
        for name in (*positive, *references, *amplitudes, "alpha", "interval_level"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")
            value = float(value)
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name in amplitudes and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name in ("alpha", "interval_level") and not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
            object.__setattr__(self, name, value)
        if not 0 < 1 - self.alpha < 1 or not 0.5 < 0.5 + self.interval_level / 2 < 1:
            raise ValueError("alpha or interval_level is too close to a boundary for float precision")


DEFAULT_CONFIG = ExperimentConfig()
N_EVALUATION = 128
N_DEVELOPMENT = 16
DEVELOPMENT_SEED_START = 0
EVALUATION_SEED_START = 10000
STATUSES = ("consistent", "identified", "ambiguous", "unexplained", "insufficient_evidence")
CASE_NAMES = (
    "no_fault", "storage_step", "storage_drift", "inflow_offset", "inflow_gain",
    "constant_flow_offset", "constant_flow_gain", "storage_common_offset",
    "legitimate_flow_change", "storage_drift_with_nuisance", "omitted_flow_with_nuisance",
    "omitted_flow_unmodeled", "storage_step_unrestricted_nuisance",
)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    level: str
    record: BalanceRecord
    raw_intervention: np.ndarray
    hypotheses: Mapping[str, np.ndarray]
    nuisance: np.ndarray | None
    magnitude: float
    magnitude_unit: str
    instrument_fault: bool
    true_hypothesis: str | None
    explanation: str


def build_case(name: str, level: str = "strong", *,
               config: ExperimentConfig = DEFAULT_CONFIG) -> SyntheticCase:
    """Build a declared noiseless record and raw per-unit intervention templates.

    Raw order is all mean storage readings, then interval-major inflow/outflow.
    Storage means are exact for constant net flow within each interval. The common
    calibration reference is one independent standard-normal draw multiplied by
    the declared per-channel sensitivities, in each channel's physical units.
    """
    if name not in CASE_NAMES or level not in ("weak", "strong"):
        raise ValueError("unknown case or amplitude level")
    n, dt, onset = config.intervals, config.interval_seconds, config.onset_interval
    if n < 8 or not 1 <= onset < n - 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("need positive intervals and an interior known onset")
    edges = np.arange(n + 1, dtype=float) * dt
    centers = (edges[:-1] + edges[1:]) / 2
    j = np.arange(n, dtype=float)
    constant = name.startswith("constant_flow")
    inflow = np.full(n, 3.0) if constant else 3.0 + 1.1 * np.sin(0.8 * j) + 0.4 * np.cos(1.7 * j)
    net = 0.15 + 0.10 * np.cos(0.5 * j)
    outflow = inflow - net
    flows = np.column_stack((inflow, outflow))
    boundary_storage = 100.0 + np.r_[0.0, np.cumsum(net * dt)]
    storage = (boundary_storage[:-1] + boundary_storage[1:]) / 2
    sigma = np.r_[np.full(n, config.storage_sd_m3),
                  np.full(2 * n, config.flow_sd_m3_per_s)]
    reference = np.r_[np.full(n, config.reference_storage_m3),
                      np.tile([config.reference_inflow_m3_per_s,
                               config.reference_outflow_m3_per_s], n)]
    covariance = np.diag(sigma ** 2) + np.outer(reference, reference)
    record = BalanceRecord(
        edges=edges, storage=storage, flows=flows, flow_signs=np.array([1.0, -1.0]),
        covariance=covariance, storage_support="mean", flow_support="mean",
        within_interval_model="constant_net_flow", volume_unit="m3", flow_unit="m3/s",
        evidence_ids=tuple(f"synthetic:{name}:{i}" for i in range(3 * n)),
    )
    H = balance_residuals(record).operator
    step = (j >= onset).astype(float)
    elapsed = np.maximum(centers - edges[onset], 0.0)
    templates = {key: np.zeros(3 * n) for key in
                 ("storage_step", "storage_drift", "inflow_offset", "inflow_gain")}
    templates["storage_step"][:n] = step
    templates["storage_drift"][:n] = elapsed
    templates["inflow_offset"][n::2] = step
    templates["inflow_gain"][n::2] = step * inflow
    hypotheses = {key: H @ value for key, value in templates.items()}
    unit, magnitude = "m3", getattr(config, f"{level}_storage_m3")
    nuisance = None
    instrument_fault = True
    true_hypothesis: str | None = name
    explanation = "One instrument, one declared fault profile; unknown unrestricted signed amplitude."
    if name in templates:
        raw = templates[name].copy()
    elif name.startswith("constant_flow"):
        true_hypothesis = "inflow_gain" if name.endswith("gain") else "inflow_offset"
        raw = templates[true_hypothesis].copy()
        explanation = "Constant inflow makes offset and gain proportional; storage drift also confounds offset."
    elif name == "storage_common_offset":
        raw = np.r_[np.ones(n), np.zeros(2 * n)]
        true_hypothesis = None
        explanation = "A storage offset present throughout the record cancels under differencing."
    elif name == "legitimate_flow_change":
        raw = templates["storage_drift"] + templates["inflow_offset"]
        true_hypothesis, instrument_fault = None, False
        explanation = "A real inflow change and its exactly conserving storage response; no instrument fault."
    elif name == "no_fault":
        raw = np.zeros(3 * n)
        true_hypothesis, instrument_fault, magnitude = None, False, 0.0
        explanation = "Conserving mean measurements with exactly the declared Gaussian noise."
    elif name == "storage_step_unrestricted_nuisance":
        raw = templates["storage_step"].copy()
        true_hypothesis = "storage_step"
        nuisance = np.eye(H.shape[0])
        explanation = "An unrestricted nuisance at every residual consumes all information; diagnosis must abstain."
    else:
        raw = templates["storage_drift"].copy()
        nuisance = hypotheses["storage_drift"][:, None] if name.endswith("with_nuisance") else None
        if name.startswith("omitted_flow"):
            instrument_fault, true_hypothesis = False, None
            explanation = "Ungauged real inflow changes storage but is absent from gauged flow: physical model discrepancy."
        else:
            true_hypothesis = "storage_drift"
            explanation = "Storage drift is indistinguishable from an unrestricted constant ungauged-flow nuisance."
    if true_hypothesis == "inflow_gain":
        unit, magnitude = "fraction", getattr(config, f"{level}_gain")
    elif (true_hypothesis in ("storage_drift", "inflow_offset") or
          name.startswith("omitted_flow") or name == "legitimate_flow_change"):
        unit, magnitude = "m3/s", getattr(config, f"{level}_rate_m3_per_s")
    return SyntheticCase(name, level, record, raw, hypotheses, nuisance, magnitude,
                         unit, instrument_fault, true_hypothesis, explanation)


def _rate(successes: int, total: int) -> dict:
    return {"count": int(successes), "denominator": int(total),
            "rate": float(successes / total) if total else None}


def _score(outcomes: list[dict], case: SyntheticCase) -> dict:
    n = len(outcomes)
    identified = [r for r in outcomes if r["status"] == "identified"]
    correctly_identified = [r for r in identified if r["correct_attribution"]]
    interval_rows = [r for r in correctly_identified if r["true_profile_interval_contains"] is not None]
    all_profile_rows = [r for r in outcomes if r["true_profile_interval_contains"] is not None]
    detected = sum(r["diagnostic_rejected_no_fault"] for r in outcomes)
    baseline_detected = sum(r["balance_only_rejected"] for r in outcomes)
    return {
        "n_records": n,
        "status_counts": {status: sum(r["status"] == status for r in outcomes) for status in STATUSES},
        "diagnostic_rejection": _rate(detected, n),
        "balance_only_rejection": _rate(baseline_detected, n),
        "instrument_fault_detection": _rate(detected if case.instrument_fault else 0,
                                             n if case.instrument_fault else 0),
        "instrument_fault_misses": n - detected if case.instrument_fault else None,
        "balance_only_instrument_fault_detection": _rate(baseline_detected if case.instrument_fault else 0,
                                                          n if case.instrument_fault else 0),
        "wrong_attribution_given_identified": _rate(sum(not r["correct_attribution"] for r in identified),
                                                    len(identified)),
        "false_instrument_attribution_on_physical_controls": _rate(len(identified) if not case.instrument_fault else 0,
                                                                  n if not case.instrument_fault else 0),
        "interval_coverage_given_correct_identification": _rate(
            sum(r["true_profile_interval_contains"] for r in interval_rows), len(interval_rows)),
        "interval_coverage_for_true_profile_regardless_of_selection": _rate(
            sum(r["true_profile_interval_contains"] for r in all_profile_rows), len(all_profile_rows)),
    }


def run_case(case: SyntheticCase, seeds: tuple[int, ...], *,
             config: ExperimentConfig = DEFAULT_CONFIG) -> dict:
    """Analyze one fixed horizon per independent record; labels enter scoring only."""
    series = balance_residuals(case.record)
    H, covariance = series.operator, series.covariance
    raw_covariance = np.asarray(case.record.covariance)
    raw_cholesky = np.linalg.cholesky(raw_covariance)
    residual_cholesky = np.linalg.cholesky(covariance)
    # Empty catalogue supplies the same reference null without candidate fitting.
    threshold = diagnose(np.zeros_like(series.residual), covariance, {},
                         alpha=config.alpha).null_threshold
    raw_noises = np.stack([np.random.default_rng(seed).standard_normal(H.shape[1]) for seed in seeds])
    residual_noises = (raw_noises @ raw_cholesky.T) @ H.T
    response = H @ case.raw_intervention
    outcomes = []
    for seed, noise in zip(seeds, residual_noises, strict=True):
        # Both signs are declared in advance, and each candidate fits an unknown sign.
        amplitude = case.magnitude * (1 if seed % 2 == 0 else -1)
        residual = series.residual + amplitude * response + noise
        result = diagnose(residual, covariance, case.hypotheses, nuisance=case.nuisance,
                          alpha=config.alpha, interval_level=config.interval_level)
        whitened = np.linalg.solve(residual_cholesky, residual)
        baseline_statistic = float(whitened @ whitened)
        true_fit = next((fit for fit in result.fits if fit.name == case.true_hypothesis), None)
        interval_contains = None
        if true_fit is not None and true_fit.interval is not None:
            interval_contains = bool(true_fit.interval[0] <= amplitude <= true_fit.interval[1])
        selected = result.candidates[0] if result.status == "identified" else None
        outcomes.append({
            "seed": seed, "true_signed_magnitude": amplitude, "magnitude_unit": case.magnitude_unit,
            "status": result.status, "candidates": list(result.candidates),
            "diagnostic_rejected_no_fault": bool(result.null_dof > 0 and result.null_statistic > result.null_threshold),
            "balance_only_rejected": bool(baseline_statistic > threshold),
            "balance_only_statistic": baseline_statistic,
            "correct_attribution": bool(case.instrument_fault and selected is not None and selected == case.true_hypothesis),
            "true_profile_interval_contains": interval_contains,
            "null_statistic": result.null_statistic,
            "fits": {fit.name: {"fit_statistic": fit.fit_statistic, "adequate": fit.adequate,
                                 "amplitude": fit.amplitude,
                                 "interval": list(fit.interval) if fit.interval is not None else None}
                     for fit in result.fits},
        })
    return {"aggregate": _score(outcomes, case), "per_seed": outcomes}


def run_benchmark(n_eval: int = N_EVALUATION, *, n_development: int = N_DEVELOPMENT,
                  config: ExperimentConfig = DEFAULT_CONFIG) -> dict:
    for name, value in (("n_eval", n_eval), ("n_development", n_development)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer, not a boolean")
    if n_development >= EVALUATION_SEED_START:
        raise ValueError("development and evaluation seed sets must be disjoint")
    development = tuple(range(DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + n_development))
    evaluation = tuple(range(EVALUATION_SEED_START, EVALUATION_SEED_START + n_eval))
    cases = {}
    for name in CASE_NAMES:
        for level in (("weak",) if name == "no_fault" else ("weak", "strong")):
            case = build_case(name, level, config=config)
            series = balance_residuals(case.record)
            reference = diagnose(np.zeros_like(series.residual), series.covariance, case.hypotheses,
                                 nuisance=case.nuisance, alpha=config.alpha,
                                 interval_level=config.interval_level)
            balance_reference = diagnose(np.zeros_like(series.residual), series.covariance, {},
                                         alpha=config.alpha)
            cases[f"{name}/{level}"] = {
                "name": name, "level": level, "magnitude": case.magnitude, "magnitude_unit": case.magnitude_unit,
                "instrument_fault": case.instrument_fault, "true_hypothesis": case.true_hypothesis,
                "explanation": case.explanation,
                "nuisance_columns": 0 if case.nuisance is None else case.nuisance.shape[1],
                "signal_norm_per_unit": float(np.linalg.norm(series.operator @ case.raw_intervention)),
                "hypothesis_units": {"storage_step": "m3", "storage_drift": "m3/s",
                                     "inflow_offset": "m3/s", "inflow_gain": "fraction"},
                "diagnostic_reference": {
                    "null_dof": reference.null_dof, "null_threshold": reference.null_threshold,
                    "fits": {fit.name: {"observable": fit.observable, "fit_dof": fit.fit_dof,
                                         "fit_threshold": fit.fit_threshold, "amplitude_sd": fit.amplitude_sd,
                                         "explanation": fit.explanation} for fit in reference.fits},
                    "note": "Fixed design and covariance make these reference quantities identical for every seed.",
                },
                "balance_only_reference": {"dof": balance_reference.null_dof,
                                           "threshold": balance_reference.null_threshold,
                                           "alpha": config.alpha},
                "development": run_case(case, development, config=config),
                "evaluation": run_case(case, evaluation, config=config),
            }
    return {
        "schema_version": "fsre-fluid-baseline-v1", "config": asdict(config), "provenance": provenance(),
        "design": {
            "scope": "Synthetic Gaussian fixed-horizon evidence only; no field validation or operational false-alarm claim.",
            "development_seeds": list(development), "evaluation_seeds": list(evaluation),
            "selection": "All noise scales, templates, amplitudes, horizon and alpha were declared before evaluation. No parameters or thresholds are fitted on either seed set.",
            "independence": "Independent noise per seed within a case; seeds reused across cases for paired comparisons. Do not treat pooled case outcomes as independent.",
            "onset": "Known onset and nominal commanded flow pattern; gain templates use that declared profile, not noisy evaluation observations.",
            "measurement": "Interval-mean storage and flows, constant net flow within each interval, full H C H' covariance including one shared Gaussian calibration reference.",
            "amplitudes": "Arbitrary declared simulation magnitudes in m3, m3/s or fractional gain, with alternating signs; not industry sensitivity targets.",
            "noise": "Independent per-reading Gaussian noise plus a shared rank-one Gaussian calibration perturbation. Faults alter means only; covariance is fixed, including for gain.",
            "event": "One test after a fixed record per seed; detection is rejection of no-fault after declared nuisance. No repeated-window or online delay claim.",
            "baseline": "Simple full-record balance-only Mahalanobis detector at the same alpha, with no nuisance or attribution. Physical omitted flow can rightly violate its gauged balance.",
            "intervals": "Candidate-conditional Gaussian intervals, not selection-adjusted. Report both true-profile coverage and coverage conditional on correct identification with explicit denominators.",
            "causal_limits": "A channel/profile explanation is conditional on the candidate and nuisance catalogue. Consistent does not mean healthy; an unmodeled physical input can mimic a fault.",
            "not_tested": ["unknown onset", "multiple faulty instruments", "uncertain gain templates", "colored or nonlinear model mismatch", "real sensor degradation", "thermal balances"],
        },
        "cases": cases,
    }


def render(report: dict) -> str:
    md = ["# Synthetic fluid measurement benchmark", "", header_line(report["provenance"]), "",
          "**Synthetic research evidence only. No field validation or operational alarm performance is established.**", "",
          f"Each case has {len(report['design']['development_seeds'])} development records and "
          f"{len(report['design']['evaluation_seeds'])} disjoint evaluation records. Configurations are fixed; "
          "neither set selects thresholds. The table uses evaluation records only.", ""]
    md.extend(f"- **{key.replace('_', ' ').capitalize()}:** {value}" for key, value in report["design"].items()
              if key not in ("development_seeds", "evaluation_seeds", "not_tested"))
    md += ["", "Not tested: " + ", ".join(report["design"]["not_tested"]) + ".", "",
           "| case / magnitude (±) | diagnostic reject | balance-only reject | consistent | identified | ambiguous | unexplained | insufficient | wrong / identified | covered / correctly identified intervals |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fraction(value: dict) -> str:
        return f"{value['count']}/{value['denominator']}" if value["denominator"] else "n/a (0)"
    for key, case in report["cases"].items():
        a = case["evaluation"]["aggregate"]
        counts = a["status_counts"]
        md.append(f"| {key}: {case['magnitude']:g} {case['magnitude_unit']} | "
                  f"{fraction(a['diagnostic_rejection'])} | {fraction(a['balance_only_rejection'])} | "
                  + " | ".join(str(counts[s]) for s in STATUSES) + " | "
                  + fraction(a["wrong_attribution_given_identified"]) + " | "
                  + fraction(a["interval_coverage_given_correct_identification"]) + " |")
    md += ["", "A missed instrument event includes a structurally invisible or nuisance-confounded fault. "
           "A rejected gauged balance during omitted real flow is not proof of a failed instrument. "
           "The JSON retains every seed, candidate fit, interval, status and scoring denominator.", "",
           "## Declared cases", ""]
    for key, case in report["cases"].items():
        if case["level"] == "weak":
            md.append(f"- **{case['name']}:** {case['explanation']}")
    md += ["", "## Frozen configuration", "", "```json", json.dumps(report["config"], indent=2), "```", ""]
    return "\n".join(md)


def main(out_dir: Path, n_eval: int = N_EVALUATION, *, n_development: int = N_DEVELOPMENT,
         quiet: bool = False) -> int:
    report = run_benchmark(n_eval, n_development=n_development)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fluid_baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render(report)
    (out_dir / "fluid_baseline.md").write_text(text, encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--n-eval", type=int, default=N_EVALUATION)
    parser.add_argument("--n-development", type=int, default=N_DEVELOPMENT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, args.n_eval, n_development=args.n_development, quiet=args.quiet))
