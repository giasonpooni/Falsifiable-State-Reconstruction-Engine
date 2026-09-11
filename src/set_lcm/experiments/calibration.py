"""In-loop null of the consistency statistic, and a threshold x debounce sweep.

The kernel test (tests/test_lcm.py) shows stat ~ chi2(1) for i.i.d. draws with
the TRUE covariance. That says nothing about the loop, where the statistic is
built from the filter's own reported P. This module measures that null
empirically over the windows where the joint hypothesis (constraint AND model
AND calibrated uncertainty) holds, and sweeps the guard's threshold and
debounce on the leak scenario so that false-alarm rate, detection and error can
be read off together instead of from one point.

    python -m set_lcm.experiments.calibration [n_seeds]   -> results/calibration.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..testbed.runner import EstimatorSpec
from .phase1 import DETECT_WITHIN, N_SEEDS, iter_runs, run_scenario

# Windows over which the joint hypothesis holds for the shipped scenarios.
NULL_WINDOWS: dict[str, tuple[int, int]] = {
    "closed_noise": (0, 600),
    "closed_blackout_pumpbias": (0, 50),
    "leak_stale_constraint": (0, 300),
    "bias_quant_delay": (0, 200),
}
THRESHOLDS = {"chi2_1(0.95)": 3.841, "chi2_1(0.99)": 6.635, "chi2_1(0.999)": 10.828}
NOMINAL = {"chi2_1(0.95)": 0.05, "chi2_1(0.99)": 0.01, "chi2_1(0.999)": 0.001}
KF = EstimatorSpec("kf", "kf", None)
SWEEP_QS = (0.95, 0.99, 0.999)
SWEEP_DEBOUNCES = (1, 3, 10)


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    den = float(x @ x)
    return float(x[:-1] @ x[1:]) / den if den > 0 else 0.0


def null_stats(n_seeds: int = N_SEEDS) -> dict:
    """Pool the unconstrained KF's consistency statistic over each null window and seed."""
    out = {}
    for name, (lo, hi) in NULL_WINDOWS.items():
        pooled, rhos = [], []
        for _, _, _, _, rr in iter_runs(name, KF, n_seeds):
            s = rr.stat[lo:hi]
            s = s[~np.isnan(s)]
            pooled.append(s)
            rhos.append(_lag1(s))
        s = np.concatenate(pooled)
        rho = float(np.mean(rhos))
        tau = (1.0 + rho) / (1.0 - rho) if rho < 1.0 else float("inf")   # AR(1) integrated autocorrelation time
        out[name] = {
            "window": [lo, hi],
            "n_seeds": n_seeds,
            "n_samples": int(s.size),
            "mean": float(s.mean()),
            "q95": float(np.quantile(s, 0.95)),
            "q99": float(np.quantile(s, 0.99)),
            "q999": float(np.quantile(s, 0.999)),
            "exceedance": {k: {"empirical": float((s > v).mean()), "nominal": NOMINAL[k]} for k, v in THRESHOLDS.items()},
            "lag1_autocorr": rho,
            "autocorr_time": tau,
            "effective_samples": float(s.size / tau) if np.isfinite(tau) else 0.0,
        }
    return out


def threshold_sweep(
    n_seeds: int = N_SEEDS,
    qs: tuple[float, ...] = SWEEP_QS,
    debounces: tuple[int, ...] = SWEEP_DEBOUNCES,
    scenario: str = "leak_stale_constraint",
    window: str = "leak_and_after",
) -> list[dict]:
    rows = []
    for q in qs:
        for d in debounces:
            spec = EstimatorSpec(f"kf+hard+guard(q={q},debounce={d})", "kf", "hard", guard=True, threshold_q=q, debounce=d)
            agg, _, _ = run_scenario(scenario, n_seeds, specs=[spec])
            a = agg[spec.name]
            rows.append({
                "q": q, "debounce": d,
                "fa_rate": a["false_alarms"]["rate"], "fa_max": a["false_alarms"]["max"],
                "detected_within": a["detection"]["detected_within"], "n": a["detection"]["n"],
                "median_delay": a["detection"]["median_delay"], "median_censored": a["detection"]["median_censored"],
                "rmse": a["rmse_by_window"][window]["mean"],
                "cov95": a["coverage95_by_window"][window]["mean"],
                "nz": a["nz_rms_by_window"][window]["mean"],
                "held": a["held_steps"]["mean"],
            })
    return rows


def render(nulls: dict, sweep: list[dict], n_seeds: int) -> str:
    md = ["# In-loop null of the consistency statistic", "",
          f"Unconstrained KF, {n_seeds} seeds, statistic r²/(A P Aᵀ) computed from the filter's own "
          "reported P over windows where the joint hypothesis holds. Under an exact χ²(1) null the mean "
          "would be 1.0 and the exceedances would equal the nominal tail probabilities. The lag-1 "
          "autocorrelation gives an AR(1) integrated autocorrelation time τ; the number of effectively "
          "independent samples is n/τ.", "",
          "| window | n | mean | q95 | q99 | q999 | P(>3.841) [0.05] | P(>6.635) [0.01] | P(>10.828) [0.001] | ρ₁ | τ | n_eff |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in nulls.items():
        e = v["exceedance"]
        md.append(f"| {name} {v['window']} | {v['n_samples']} | {v['mean']:.3f} | {v['q95']:.2f} | {v['q99']:.2f} | "
                  f"{v['q999']:.2f} | {e['chi2_1(0.95)']['empirical']:.4f} | {e['chi2_1(0.99)']['empirical']:.4f} | "
                  f"{e['chi2_1(0.999)']['empirical']:.5f} | {v['lag1_autocorr']:.3f} | {v['autocorr_time']:.1f} | "
                  f"{v['effective_samples']:.0f} |")
    md += ["", "# Threshold × debounce sweep (kf+hard+guard, leak_stale_constraint)", "",
           f"Pre-onset false-alarm rate per step, seeds detected within {DETECT_WITHIN} steps of onset, censored "
           "median delay, and (RMSE, cov95, nz) over the leak window. The shipped setting is q = 0.999, "
           "debounce = 3.", "",
           "| q | debounce | FA rate | FA max | detected | median delay | RMSE leak | cov95 leak | nz leak | held |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sweep:
        med = f"> {r['median_delay']:.0f}" if r["median_censored"] else f"{r['median_delay']:.0f}"
        md.append(f"| {r['q']} | {r['debounce']} | {r['fa_rate']:.1e} | {r['fa_max']} | {r['detected_within']}/{r['n']} | "
                  f"{med} | {r['rmse']:.2f} | {r['cov95']:.2f} | {r['nz']:.2f} | {r['held']:.0f} |")
    return "\n".join(md) + "\n"


def main(out_dir: Path, n_seeds: int = N_SEEDS, *, quiet: bool = False) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    nulls = null_stats(n_seeds)
    sweep = threshold_sweep(n_seeds)
    text = render(nulls, sweep, n_seeds)
    (out_dir / "calibration.md").write_text(text, encoding="utf-8")
    (out_dir / "calibration.json").write_text(
        json.dumps({"n_seeds": n_seeds, "null": nulls, "threshold_sweep": sweep}, indent=2), encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path.cwd() / "results", int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS))
