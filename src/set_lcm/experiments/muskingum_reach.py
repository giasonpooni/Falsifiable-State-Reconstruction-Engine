"""Roadmap stage 3: a second independent balance, built with the rows that make it worth building.

The design study (experiments/second_balance.py) predicted, from the algebra alone, what this
topology could separate: continuity alone isolates almost nothing, and Muskingum's own storage
relation takes it to 5 of 7 declared candidates. Those were predictions about a system that did
not exist. This builds it and tests them against simulated records.

The decision rule is the residual VECTOR and its joint covariance, through
diagnostics.diagnose() -- not a rank calculation and not a scalar threshold, which the roadmap
explicitly excludes. Each interval contributes its declared rows, adjacent intervals share a
storage reading, and Cov(r) = M R M^T carries that sharing exactly rather than pretending the
intervals are independent.

WHAT IS SCORED. Faults are injected into a hidden truth and the estimator side sees readings
only. For each injected case the report gives what the rule said, against explicit
denominators: identified-and-correct, identified-but-wrong, ambiguous, unexplained, and
consistent (a miss). A wrong identification is counted separately from a miss because they
fail differently.

FOUR CONTRASTS, each isolating one thing:

    rows            continuity only against continuity + routing, same records, same rule
    catalogue       whether the PHYSICAL explanation is a declared candidate
    parameters      declared K and x equal to the truth, or in error
    A_var           that same parameter error, declared through ConstraintSet.A_var

The third and fourth are why stage 3b existed: routing rows carry measured parameters, and the
isolation they buy is conditional on parameters the kernel could not carry uncertainty for
until A_var did.

    python -m set_lcm.experiments.muskingum_reach  -> results/muskingum_reach.{md,json}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..diagnostics import diagnose
from ..testbed.muskingum import (
    Fault, MuskingumConfig, constraint_rows, design_matrix, fault_reading_direction, observe,
    reading_names, simulate,
)
from .provenance import REPO_ROOT, header_line, provenance

N_STEPS = 21                     # 20 intervals
ALPHA = 0.01
ONSET = 6                        # declared, and supplied to the rule: no onset search here

DECLARED = MuskingumConfig(K=1.0, x=0.2, dt=1.0)
TRUE_WITH_PARAMETER_ERROR = MuskingumConfig(K=1.15, x=0.28, dt=1.0)   # the river, not the model

SIGMA = {"storage_1": 0.20, "storage_2": 0.20,
         "inflow_gauge": 0.30, "middle_gauge": 0.30, "outflow_gauge": 0.30}

DEVELOPMENT_SEEDS = tuple(range(16))
EVALUATION_SEEDS = tuple(range(10_000, 10_064))

# Declared instrument candidates. The physical explanation is declared separately, because
# whether it belongs in the catalogue is one of the things this experiment measures.
INSTRUMENT_CANDIDATES = {
    "storage 1 sensor bias": ("storage_1",),
    "storage 2 sensor bias": ("storage_2",),
    "inflow gauge bias": ("inflow_gauge",),
    "middle gauge bias": ("middle_gauge",),
    "outflow gauge bias": ("outflow_gauge",),
    "common flow-gauge drift": ("inflow_gauge", "middle_gauge", "outflow_gauge"),
}
PHYSICAL_CANDIDATE = "ungauged lateral inflow"

# Two locked magnitudes per case, in multiples of the affected sensor's declared sigma.
MAGNITUDE_MULTIPLES = (1.5, 4.0)


def injected_cases() -> list[tuple[str, Fault, str | None]]:
    """(label, what is injected, the candidate that would be the right answer)."""
    cases: list[tuple[str, Fault, str | None]] = [("healthy", Fault("none"), None)]
    for name, sensors in INSTRUMENT_CANDIDATES.items():
        kind = "common_drift" if len(sensors) > 1 else "sensor_bias"
        cases.append((name, Fault(kind, 0.0, ONSET, sensors), name))
    cases.append((PHYSICAL_CANDIDATE,
                  Fault("lateral_inflow", 0.0, ONSET, (), physical=True), PHYSICAL_CANDIDATE))
    return cases


def _magnitude(fault: Fault, multiple: float) -> float:
    if fault.kind == "lateral_inflow":
        return multiple * SIGMA["inflow_gauge"]
    if not fault.sensors:
        return 0.0
    return multiple * min(SIGMA[name] for name in fault.sensors)


def candidates(n_steps: int, *, include_physical: bool) -> dict[str, np.ndarray]:
    """Declared signatures on the readings, before the residual operator is applied."""
    declared = {name: fault_reading_direction(name, sensors, ONSET, n_steps)
                for name, sensors in INSTRUMENT_CANDIDATES.items()}
    if include_physical:
        # An ungauged inflow is not a reading perturbation. It is declared through the reading
        # whose ABSENCE it corrupts: the inflow gauge, which never sees the water. That is the
        # same direction the inflow gauge's own bias produces, which is the point.
        declared[PHYSICAL_CANDIDATE] = fault_reading_direction(
            PHYSICAL_CANDIDATE, ("inflow_gauge",), ONSET, n_steps)
    return declared


def run_one(*, truth_config: MuskingumConfig, declared_config: MuskingumConfig, fault: Fault,
            seed: int, with_routing: bool, include_physical: bool,
            declare_parameter_uncertainty: bool) -> dict:
    """One record through the rule. The estimator side never receives the truth object."""
    truth = simulate(truth_config, N_STEPS, fault)
    readings = observe(truth, SIGMA, np.random.default_rng(seed))
    y, R = readings.stacked()

    M = design_matrix(declared_config, N_STEPS, with_routing=with_routing)
    residual = M @ y
    covariance = M @ R @ M.T
    if declare_parameter_uncertainty:
        covariance = covariance + _parameter_covariance(declared_config, y, N_STEPS,
                                                        with_routing=with_routing)
    signatures = {name: M @ direction
                  for name, direction in candidates(N_STEPS, include_physical=include_physical).items()}
    result = diagnose(residual, covariance, signatures, alpha=ALPHA)
    return {"status": result.status, "candidates": list(result.candidates),
            "min_separation": result.min_separation,
            "null_statistic": result.null_statistic, "null_threshold": result.null_threshold}


def _parameter_covariance(config: MuskingumConfig, y: np.ndarray, n_steps: int, *,
                          with_routing: bool) -> np.ndarray:
    """Cov(E y) for declared uncertainty in the routing parameters K and x.

    Only the routing rows carry parameters; continuity carries dt and signs. Perturbing K and x
    moves the two routing coefficients, so the row error is a linear map of (dK, dx) and the
    contribution is the quadratic form that map induces on the readings -- the same
    state-dependent term ConstraintSet.A_var declares for a single interval, assembled here for
    the whole record because the parameters are SHARED across every interval rather than drawn
    afresh at each one.
    """
    if not with_routing:
        return np.zeros((design_matrix(config, n_steps, with_routing=False).shape[0],) * 2)
    step = 1e-6
    sensitivities = []
    for field, value in (("K", config.K), ("x", config.x)):
        bumped = MuskingumConfig(**{**{"K": config.K, "x": config.x, "dt": config.dt},
                                    field: value + step})
        derivative = (design_matrix(bumped, n_steps, with_routing=True)
                      - design_matrix(config, n_steps, with_routing=True)) / step
        sensitivities.append(derivative @ y)
    J = np.column_stack(sensitivities)
    return J @ np.diag([PARAMETER_SIGMA["K"] ** 2, PARAMETER_SIGMA["x"] ** 2]) @ J.T


# Declared uncertainty of the routing parameters. A calibrated reach reports K and x with
# error; these are the declared standard deviations, not a fit to this record.
PARAMETER_SIGMA = {"K": 0.15, "x": 0.08}


def score(cases, seeds, **kwargs) -> dict:
    """Every injected case over every seed, with explicit denominators."""
    out = {}
    for label, template, correct in cases:
        for multiple in (MAGNITUDE_MULTIPLES if template.kind != "none" else (0.0,)):
            fault = Fault(template.kind, _magnitude(template, multiple), template.onset,
                          template.sensors, template.physical)
            outcomes, statuses = Counter(), Counter()
            for seed in seeds:
                result = run_one(fault=fault, seed=seed, **kwargs)
                statuses[result["status"]] += 1
                if template.kind == "none":
                    outcomes["correctly quiet" if result["status"] == "consistent"
                             else "false alarm"] += 1
                elif result["status"] == "consistent":
                    outcomes["missed"] += 1
                elif result["status"] == "identified":
                    outcomes["identified correctly" if result["candidates"] == [correct]
                             else "identified WRONG"] += 1
                elif result["status"] == "ambiguous":
                    outcomes["ambiguous, correct among them" if correct in result["candidates"]
                             else "ambiguous, correct absent"] += 1
                else:
                    outcomes[result["status"]] += 1
            key = label if template.kind == "none" else f"{label} @ {multiple:g} sigma"
            out[key] = {"injected": label, "kind": template.kind,
                        "magnitude_multiple": multiple,
                        "magnitude": float(fault.magnitude),
                        "correct_candidate": correct, "n_seeds": len(seeds),
                        "outcomes": dict(outcomes), "statuses": dict(statuses)}
    return out


def compute() -> dict:
    cases = injected_cases()
    common = dict(truth_config=DECLARED, declared_config=DECLARED, include_physical=False,
                  declare_parameter_uncertainty=False)
    variants = {
        "continuity only": {**common, "with_routing": False},
        "continuity + routing": {**common, "with_routing": True},
        "continuity + routing, physical candidate declared":
            {**common, "with_routing": True, "include_physical": True},
        "routing with parameter error, A treated as exact":
            {**common, "with_routing": True, "include_physical": True,
             "truth_config": TRUE_WITH_PARAMETER_ERROR},
        "routing with parameter error, A_var declared":
            {**common, "with_routing": True, "include_physical": True,
             "truth_config": TRUE_WITH_PARAMETER_ERROR, "declare_parameter_uncertainty": True},
    }
    development = {name: score(cases, DEVELOPMENT_SEEDS, **kwargs) for name, kwargs in variants.items()}
    evaluation = {name: score(cases, EVALUATION_SEEDS, **kwargs) for name, kwargs in variants.items()}
    out = {
        "schema_version": "fsre-muskingum-reach-v1",
        "scope": "Simulated two-reach Muskingum records with known injected faults. Synthetic "
                 "throughout: no field validation and no real record.",
        "provenance": {"generation": provenance()},
        "declared": {
            "config": {"K": DECLARED.K, "x": DECLARED.x, "dt": DECLARED.dt},
            "true_config_under_parameter_error": {"K": TRUE_WITH_PARAMETER_ERROR.K,
                                                  "x": TRUE_WITH_PARAMETER_ERROR.x,
                                                  "dt": TRUE_WITH_PARAMETER_ERROR.dt},
            "parameter_sigma": dict(PARAMETER_SIGMA),
            "sigma": dict(SIGMA), "n_steps": N_STEPS, "alpha": ALPHA, "onset": ONSET,
            "magnitude_multiples": list(MAGNITUDE_MULTIPLES),
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "onset_status": "declared and supplied to the rule; no onset search is performed",
            "hydrograph_status": "deterministic, so a seed changes the readings and never the river",
        },
        "structure": _structure(),
        "development": development,
        "evaluation": evaluation,
        "limitations": [
            "Synthetic records from the same Muskingum model the rows declare, except where parameter error is injected on purpose. Agreement is not field validation.",
            "One hydrograph, one reach pair, one onset, two magnitudes. A different wave or onset could order these differently.",
            "The onset is declared and supplied; searching for it is a different and harder problem, and the amplitude intervals diagnose() reports are conditional on the supplied profile.",
            "An ungauged lateral inflow and an inflow-gauge bias produce the same residual direction. No covariance and no amount of record separates them; only another gauge does.",
            "A coordinated offset of both storage sensors by K times a common offset on all three flow gauges is in the null space and produces exactly zero residual, at any magnitude.",
            "Parameter uncertainty is declared as a linear sensitivity in K and x about the declared values, first order in the parameter error like the A_var term it mirrors.",
        ],
    }
    out["claims"] = claims(out)
    return out


def _structure() -> dict:
    """What each declared direction does to one interval's rows: the table behind everything."""
    labels = ["S1_start", "S1_end", "S2_start", "S2_end", "inflow", "middle", "outflow"]
    directions = {
        "storage 1 sensor bias": [1, 1, 0, 0, 0, 0, 0],
        "storage 2 sensor bias": [0, 0, 1, 1, 0, 0, 0],
        "inflow gauge bias": [0, 0, 0, 0, 1, 0, 0],
        "middle gauge bias": [0, 0, 0, 0, 0, 1, 0],
        "outflow gauge bias": [0, 0, 0, 0, 0, 0, 1],
        "common flow-gauge drift": [0, 0, 0, 0, 1, 1, 1],
        PHYSICAL_CANDIDATE: [0, 0, 0, 0, 1, 0, 0],
        "coordinated storage/flow offset (K:1)": [DECLARED.K, DECLARED.K, DECLARED.K,
                                                  DECLARED.K, 1, 1, 1],
    }
    out = {"state_order": labels, "rows": {}}
    for with_routing, key in ((False, "continuity only"), (True, "continuity + routing")):
        A = constraint_rows(DECLARED, with_routing=with_routing)
        out["rows"][key] = {name: [float(v) for v in A @ np.array(d, dtype=float)]
                            for name, d in directions.items()}
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition."""
    evaluation = r["evaluation"]

    def correct(variant, case):
        cell = evaluation[variant][case]
        return cell["outcomes"].get("identified correctly", 0) / cell["n_seeds"]

    strong = f" @ {max(MAGNITUDE_MULTIPLES):g} sigma"
    instruments = [name + strong for name in INSTRUMENT_CANDIDATES]
    bare, routed = "continuity only", "continuity + routing"
    with_physical = "continuity + routing, physical candidate declared"
    exact_A = "routing with parameter error, A treated as exact"
    declared_A = "routing with parameter error, A_var declared"
    physical_case = PHYSICAL_CANDIDATE + strong
    structure = r["structure"]["rows"]["continuity + routing"]
    return {
        "routing recovers every declared instrument fault at the strong magnitude": all(
            correct(routed, case) == 1.0 for case in instruments),
        "continuity alone recovers strictly fewer of them": (
            sum(correct(bare, case) for case in instruments)
            < sum(correct(routed, case) for case in instruments)),
        "continuity alone misses the storage sensors entirely": all(
            correct(bare, name + strong) == 0.0
            for name in ("storage 1 sensor bias", "storage 2 sensor bias")),
        "neither variant raises a false alarm on a healthy river": all(
            evaluation[variant]["healthy"]["outcomes"].get("false alarm", 0)
            <= 0.05 * evaluation[variant]["healthy"]["n_seeds"]
            for variant in (bare, routed)),
        "an undeclared physical cause is confidently blamed on an instrument": (
            evaluation[routed][physical_case]["outcomes"].get("identified WRONG", 0)
            > 0.5 * evaluation[routed][physical_case]["n_seeds"]),
        "declaring the physical cause converts that into reported ambiguity": (
            evaluation[with_physical][physical_case]["outcomes"].get("ambiguous, correct among them", 0)
            > 0.5 * evaluation[with_physical][physical_case]["n_seeds"]),
        "parameter error treated as exact raises false alarms on a healthy river": (
            evaluation[exact_A]["healthy"]["outcomes"].get("false alarm", 0)
            > 0.5 * evaluation[exact_A]["healthy"]["n_seeds"]),
        "declaring the parameter uncertainty removes them": (
            evaluation[declared_A]["healthy"]["outcomes"].get("false alarm", 0)
            < evaluation[exact_A]["healthy"]["outcomes"].get("false alarm", 0)),
        "the coordinated offset produces exactly no residual": all(
            value == 0.0 for value in structure["coordinated storage/flow offset (K:1)"]),
        "the physical cause and the inflow gauge produce the same residual": (
            structure[PHYSICAL_CANDIDATE] == structure["inflow gauge bias"]),
        "the storage sensors move only the routing rows": all(
            structure[name][:2] == [0.0, 0.0]
            for name in ("storage 1 sensor bias", "storage 2 sensor bias")),
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    declared, evaluation = report["declared"], report["evaluation"]
    strong = f" @ {max(declared['magnitude_multiples']):g} sigma"
    A("# A second independent balance: the two-reach Muskingum design")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A("The design study predicted from the algebra that continuity alone would isolate almost "
      "nothing and that Muskingum's own storage relation would take this topology to 5 of 7 "
      "declared candidates. This builds the system and tests that against simulated records. "
      "The rule is the residual vector and its joint covariance through `diagnose()`; adjacent "
      "intervals share a storage reading and `Cov(r) = M R Mᵀ` carries that exactly.")
    A("")

    A("## What each fault does to one interval's rows")
    A("")
    A("| declared direction | continuity 1 | continuity 2 | routing 1 | routing 2 |")
    A("|---|---:|---:|---:|---:|")
    for name, values in report["structure"]["rows"]["continuity + routing"].items():
        A(f"| {name} | " + " | ".join(f"{v:+.2f}" for v in values) + " |")
    A("")
    A("Three things are visible in that table alone. **The storage sensors move only the "
      "routing rows** — continuity differences the two storage readings, so a persistent bias "
      "cancels exactly, which is why continuity alone cannot see them at any magnitude. "
      "**An ungauged lateral inflow produces the same residual as an inflow-gauge bias**, so "
      "no covariance and no length of record separates them. And **a coordinated offset of "
      "both storage sensors by K times a common offset on all three flow gauges produces "
      "exactly zero**: the topology's blind spot, named.")
    A("")

    A("## Recovery, on the evaluation seeds")
    A("")
    A(f"{len(declared['evaluation_seeds'])} evaluation seeds, disjoint from the "
      f"{len(declared['development_seeds'])} development seeds, at both declared magnitudes. "
      "Cells are the fraction of records where the rule identified the injected fault and "
      "named it correctly; a wrong name is counted separately in the JSON and is not in these "
      "cells.")
    A("")
    weak_multiple, strong_multiple = min(declared["magnitude_multiples"]), max(declared["magnitude_multiples"])
    weak = f" @ {weak_multiple:g} sigma"
    variants = ["continuity only", "continuity + routing"]
    A(f"| injected fault | continuity only, {weak_multiple:g}σ | routing, {weak_multiple:g}σ | "
      f"continuity only, {strong_multiple:g}σ | routing, {strong_multiple:g}σ |")
    A("|---|---:|---:|---:|---:|")
    for name in INSTRUMENT_CANDIDATES:
        row = []
        for suffix in (weak, strong):
            for variant in variants:
                cell = evaluation[variant][name + suffix]
                row.append(f"{cell['outcomes'].get('identified correctly', 0) / cell['n_seeds']:.0%}")
        A(f"| {name} | " + " | ".join(row) + " |")
    healthy = [f"{evaluation[v]['healthy']['outcomes'].get('false alarm', 0) / evaluation[v]['healthy']['n_seeds']:.0%}"
               for v in variants]
    A(f"| *healthy river (false alarms, α = {declared['alpha']:g})* | " + healthy[0] + " | " +
      healthy[1] + " | " + healthy[0] + " | " + healthy[1] + " |")
    A("")
    A("Routing is what makes the design worth building, and the storage rows are where the "
      f"difference lives: continuity cannot see either storage sensor or a common flow drift "
      f"at {strong_multiple:g} sigma any more than at {weak_multiple:g}, because it cannot see "
      "them at all. The weaker column is the one that says this design is not magic — at "
      f"{weak_multiple:g} sigma it names the right instrument in a minority of records.")
    A("")

    A("## A physical cause the catalogue does not contain")
    A("")
    physical = PHYSICAL_CANDIDATE + strong
    without = evaluation["continuity + routing"][physical]["outcomes"]
    with_it = evaluation["continuity + routing, physical candidate declared"][physical]["outcomes"]
    n = evaluation["continuity + routing"][physical]["n_seeds"]
    A(f"An ungauged lateral inflow is injected: the river changes and every instrument is "
      f"honest. With only instrument candidates declared, the rule reports "
      f"**identified — and names an instrument — in {without.get('identified WRONG', 0) / n:.0%} "
      f"of seeds**. It is not wrong about the evidence; the two directions are identical. It is "
      f"wrong because the catalogue offered it no other answer.")
    A("")
    A(f"Declaring the physical explanation as a candidate converts that into "
      f"**{with_it.get('ambiguous, correct among them', 0) / n:.0%} ambiguous with the correct "
      f"explanation among them**. The lesson is about the catalogue, not the topology: a "
      "catalogue of only instrument faults will confidently name an instrument for a river.")
    A("")

    A("## Routing rows carry measured parameters")
    A("")
    exact_A = evaluation["routing with parameter error, A treated as exact"]["healthy"]
    declared_A = evaluation["routing with parameter error, A_var declared"]["healthy"]
    n = exact_A["n_seeds"]
    A(f"The isolation above is conditional on K and x. Running the same healthy river with the "
      f"true reach at K = {declared['true_config_under_parameter_error']['K']:g}, "
      f"x = {declared['true_config_under_parameter_error']['x']:g} while the rows still declare "
      f"K = {declared['config']['K']:g}, x = {declared['config']['x']:g}:")
    A("")
    A("| declaration | false-alarm rate on a healthy river |")
    A("|---|---:|")
    A(f"| parameter error, A treated as exact | **{exact_A['outcomes'].get('false alarm', 0) / n:.0%}** |")
    A(f"| parameter error, uncertainty declared | {declared_A['outcomes'].get('false alarm', 0) / n:.0%} |")
    A("")
    A("A mis-declared reach reads as a fault in every record until the uncertainty in its "
      "parameters is declared. That is stage 3b's `A_var` doing the job it was built for, on "
      "the first topology that needed it.")
    A("")

    A("## What this does not show")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "muskingum_reach.md").write_text(text, encoding="utf-8")
    (out_dir / "muskingum_reach.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
