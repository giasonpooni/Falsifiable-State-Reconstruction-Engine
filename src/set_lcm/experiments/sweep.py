"""Fault-magnitude sweep.

Where does the guard's dead band start, at what fault size does hard projection
begin to hurt, and what does a principled soft mode buy when the declared total
is itself uncertain?

Axes (each point: 20 seeds, same SEED / SEED_STRIDE as the Phase 1 grid):
  declared_total_error   the declared total b is off by delta kg (closed_noise otherwise); adds
                         kf+hard(b_var=0.25) and its guard, whose constraint declares sigma_b = 0.5 kg
                         (the scale of the guard's dead band, declared honestly)
  sensor_bias            undeclared sensor-1 bias of b kg from step 200 (bias_quant_delay otherwise)
  leak_rate              leak of r kg/s from reservoir 2 over steps 300-500
  uncertain_total        b = total0 + N(0, sigma_b^2) drawn per seed; adds kf+soft(lam = 1/sigma_b^2),
                         the pseudo-measurement whose variance IS the declared uncertainty, on the exact
                         set, and kf+hard(b_var) and its guard on the same b with b_var = sigma_b^2
                         declared on the ConstraintSet

Every spec on a point sees the same truth and observations; what differs is the
ConstraintSet it is handed (exact, or with b_var) and, where the declaration changes
which hypothesis is being tested, its onset: on uncertain_total the declared-b_var specs
test "b within its declared uncertainty", which holds, so their flags are false alarms.

    python -m set_lcm.experiments.sweep [n_seeds]   -> results/sweep.{md,json}
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..schema import ConstraintSet
from ..testbed.degrade import observe
from ..testbed.evaluate import evaluate
from ..testbed.runner import EstimatorSpec
from ..testbed.simulator import simulate
from .phase1 import (
    DETECT_WITHIN, N_SEEDS, SCENARIOS, SEED, SEED_STRIDE, Scenario, aggregate, declared_prior, run_spec,
)

KF = EstimatorSpec("kf", "kf", None)
HARD = EstimatorSpec("kf+hard", "kf", "hard")
GUARD = EstimatorSpec("kf+hard+guard", "kf", "hard", guard=True)
AUG = EstimatorSpec("kf_aug", "kf_aug", None)     # never projects; alpha and L are outputs with flags
CLOSEDQ = EstimatorSpec("kf_closedq", "kf_closedq", None)   # closure in Q, no constraint row, never projects
FB_GUARD = EstimatorSpec("kf+hard+fb+guard", "kf", "hard", guard=True, feedback=True)
SOFT_NAME = "kf+soft(lam=1/sigma_b^2)"
# Handed a ConstraintSet that declares b_var (the specs themselves are plain kf+hard; the
# kernel honours b_var because it is on the set).
HARD_BV = EstimatorSpec("kf+hard(b_var)", "kf", "hard")
GUARD_BV = EstimatorSpec("kf+hard(b_var)+guard", "kf", "hard", guard=True)
DTE_B_VAR = 0.25     # declared_total_error: sigma_b = 0.5 kg, the scale of the exact guard's dead band
HARD_BV_DTE = EstimatorSpec(f"kf+hard(b_var={DTE_B_VAR:g})", "kf", "hard")
GUARD_BV_DTE = EstimatorSpec(f"kf+hard(b_var={DTE_B_VAR:g})+guard", "kf", "hard", guard=True)

AXES: dict[str, dict] = {
    "declared_total_error": {
        "base": "closed_noise", "points": (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0), "unit": "kg",
        "window": "steady", "specs": (KF, HARD, GUARD, FB_GUARD, AUG, CLOSEDQ),
        "note": "The declared total is wrong by delta. The joint hypothesis is false from step 0 for delta > 0. "
                "kf_aug and kf_closedq read the constraint only through its consistency flag and never project, "
                "so their rows are the same at every delta. kf+hard+fb+guard feeds each applied projection back "
                "into the filter and stops while the guard holds.",
        "declared_specs": (HARD_BV_DTE, GUARD_BV_DTE), "declared_b_var": DTE_B_VAR,
        "declared_note": f"kf+hard(b_var={DTE_B_VAR:g}) and its guard get the same wrong total with b_var = "
                         f"{DTE_B_VAR:g} kg² declared (σ_b = 0.5 kg, the scale of the exact guard's dead band): "
                         "hard becomes the Kalman update with that pseudo-measurement variance and the guard "
                         "tests rᵀ(APAᵀ + σ_b²)⁻¹r. A fixed delta is not a draw from N(0, σ_b²), so the onset "
                         "stays at step 0 for delta > 0 and their flags count as detections.",
    },
    "sensor_bias": {
        "base": "bias_quant_delay", "points": (0.5, 1.0, 2.0, 3.0), "unit": "kg",
        "window": "post_bias", "specs": (KF, HARD, GUARD),
        "note": "Undeclared sensor-1 bias from step 200, with 0.5 kg quantization and 5-step delay.",
    },
    "leak_rate": {
        "base": "leak_stale_constraint", "points": (0.005, 0.01, 0.02, 0.05), "unit": "kg/s",
        "window": "leak_and_after", "specs": (KF, HARD, GUARD, AUG), "aug_flag": "L",
        "note": "Leak from reservoir 2 over steps 300-500; total lost = 200 x rate. kf_aug L det = seeds whose "
                f"L flag (|L̂| / σ_L > 3.29, 3 consecutive reports) fired within {DETECT_WITHIN} steps of onset.",
    },
    "uncertain_total": {
        "base": "closed_noise", "points": (0.1, 0.3, 0.5, 1.0, 2.0), "unit": "kg (sigma_b)",
        "window": "steady", "specs": (KF, HARD, GUARD),   # + soft with lam = 1/sigma_b^2, added per point
        "note": "b = total0 + N(0, sigma_b^2) per seed. The guard tests the constraint as if exact; "
                "the soft variant uses the declared uncertainty as the pseudo-measurement variance.",
        "declared_specs": (HARD_BV, GUARD_BV), "declared_b_var": "sigma_b^2",
        "declared_note": "kf+hard(b_var) and its guard get the same b with b_var = sigma_b² declared on the "
                         "ConstraintSet: hard is the Kalman update with that pseudo-measurement variance (the "
                         "same estimate as the soft variant, which carries it as λ on an exact set) and the "
                         "guard tests rᵀ(APAᵀ + σ_b²)⁻¹r, whose hypothesis -- b within its declared uncertainty "
                         "-- holds; their flags are therefore false alarms (onset none; det cells read "
                         "'FA max / rate'). kf+hard and kf+hard+guard keep treating b as exact, for contrast.",
    },
}


def _point_config(axis: str, value: float, i: int, base: Scenario):
    """Return (sim, deg, runs) for one sweep point and seed index; runs is a list of
    (spec, cs_fn, onset) with cs_fn(truth) -> the ConstraintSet that spec is handed."""
    sim = replace(base.sim, seed=base.sim.seed + SEED_STRIDE * i)
    deg = replace(base.deg, seed=base.deg.seed + SEED_STRIDE * i)
    cfg = AXES[axis]
    specs = list(cfg["specs"])
    b_var = None                 # declared b_var for cfg["declared_specs"], if the axis has them
    declared_onset = None
    if axis == "declared_total_error":
        onset = None if value == 0.0 else 0
        total = lambda truth: truth.total0 + value
        cs_fn = lambda truth: _cs(total(truth))
        b_var, declared_onset = cfg["declared_b_var"], onset
    elif axis == "sensor_bias":
        deg = replace(deg, bias=(0, float(value), 200))
        onset = 200
        cs_fn = lambda truth: _cs(truth.total0)
    elif axis == "leak_rate":
        sim = replace(sim, leak_rate=float(value))
        onset = 300
        cs_fn = lambda truth: _cs(truth.total0)
    elif axis == "uncertain_total":
        rng = np.random.default_rng(SEED + 7 + SEED_STRIDE * i)
        offset = float(rng.normal(0.0, value))
        onset = 0
        total = lambda truth: truth.total0 + offset
        cs_fn = lambda truth: _cs(total(truth), b_std=value)
        specs.append(EstimatorSpec(SOFT_NAME, "kf", "soft", lam=1.0 / value ** 2))
        b_var, declared_onset = value ** 2, None   # b IS within its declared uncertainty
    else:
        raise KeyError(axis)
    runs = [(s, cs_fn, onset) for s in specs]
    if b_var is not None:
        declared_fn = lambda truth: _cs(total(truth), b_var=b_var)
        runs += [(s, declared_fn, declared_onset) for s in cfg["declared_specs"]]
    return sim, deg, runs


def _cs(total: float, b_std: float | None = None, b_var: float | None = None) -> ConstraintSet:
    """The sweep's sum constraint. b_std only annotates the description of an EXACT set (the
    uncertain_total axis's exact variants); b_var is declared on the set itself."""
    desc = f"m1 + m2 = {total:.4g} kg" + (f" (declared std {b_std:g} kg)" if b_std else "")
    if b_var is None:
        return ConstraintSet("closed-boundary-sweep", np.array([[1.0, 1.0]]), np.array([total]), desc,
                             row_units=("kg",))
    return ConstraintSet("closed-boundary-sweep-bvar", np.array([[1.0, 1.0]]), np.array([total]),
                         desc + f", declared b_var = {b_var:g} kg^2", b_var=np.array([float(b_var)]),
                         row_units=("kg",))


def run_point(axis: str, value: float, n_seeds: int = N_SEEDS) -> dict:
    base = SCENARIOS[AXES[axis]["base"]]
    per_seed = []
    for i in range(n_seeds):
        sim, deg, runs = _point_config(axis, value, i, base)
        truth = simulate(sim)
        obs = observe(truth, deg)
        m0, m0_std = declared_prior(base, sim)
        seed_metrics = {}
        for spec, cs_fn, onset in runs:
            cs = cs_fn(truth)
            seed_metrics[spec.name] = evaluate(run_spec(truth, obs, cs, spec, m0, m0_std), truth, onset, base.windows,
                                               cs=cs)
        per_seed.append(seed_metrics)
    return aggregate(per_seed)


def run_axis(axis: str, n_seeds: int = N_SEEDS, points=None) -> dict[float, dict]:
    pts = AXES[axis]["points"] if points is None else tuple(points)
    return {float(p): run_point(axis, float(p), n_seeds) for p in pts}


def _cell(a: dict, window: str) -> str:
    return f"{a['rmse_by_window'][window]['mean']:.2f} / {a['coverage95_by_window'][window]['mean']:.2f} / {a['nz_rms_by_window'][window]['mean']:.2f}"


def render(results: dict[str, dict[float, dict]], n_seeds: int) -> str:
    md = ["# Fault-magnitude sweep", "",
          f"{n_seeds} seeds per point. Cells are (RMSE kg / cov95 / nz) over the fault window; "
          f"det = seeds flagged within {DETECT_WITHIN} steps of onset, or 'FA max / rate' (worst-seed count and "
          "mean per-step rate of flags) where the spec's hypothesis holds for the whole run; held = mean steps "
          "the guard reported model_inconsistent ('guard held' is kf+hard+guard's). nz = RMS normalised error "
          "(1.0 calibrated, >1 over-confident). Specs named (b_var…) are handed a ConstraintSet that declares "
          "that variance on b.", ""]
    for axis, pts in results.items():
        cfg = AXES[axis]
        w = cfg["window"]
        first = next(iter(pts.values()))
        specs = list(first.keys())
        det_specs = [s for s in specs if "guard" in s or "soft" in s]
        held_specs = [s for s in specs if "guard" in s and s != "kf+hard+guard"]
        # an augmented estimator gets a flag column on axes that name which flag is the fault's
        aug_flag = cfg.get("aug_flag")
        aug_specs = [s for s in specs if aug_flag and first[s].get("aug") is not None]
        det_heads = [f"{s} det" for s in det_specs] + [f"{s} {aug_flag} det" for s in aug_specs]
        held_heads = ["guard held"] + [f"{s} held" for s in held_specs]
        note = cfg["note"] + (" " + cfg["declared_note"] if "declared_note" in cfg else "")
        md += [f"## {axis}", "", note, "",
               "| " + f"{axis} [{cfg['unit']}]" + " | " + " | ".join(f"{s} (RMSE/cov/nz)" for s in specs) +
               " | " + " | ".join(det_heads + held_heads) + " |",
               "| " + " | ".join(["---"] * (1 + len(specs) + len(det_heads) + len(held_heads))) + " |"]
        for p, agg in pts.items():
            cells = [f"{p:g}"] + [_cell(agg[s], w) for s in specs]
            dets = [_det(agg[s]) for s in det_specs]
            dets += [_det_flag(agg[s]["aug"]["flags"][aug_flag]) for s in aug_specs]
            held = [f"{agg['kf+hard+guard']['held_steps']['mean']:.0f}" if "kf+hard+guard" in agg else "—"]
            held += [f"{agg[s]['held_steps']['mean']:.0f}" for s in held_specs]
            md.append("| " + " | ".join(cells + dets + held) + " |")
        md.append("")
    return "\n".join(md)


def _det(a: dict) -> str:
    """Timely detections for a spec's constraint flag, or its false alarms where no onset applies."""
    return _det_flag({"detection": a["detection"], "false_alarms": a["false_alarms"]})


def _det_flag(f: dict) -> str:
    d = f["detection"]
    if d["applicable"]:
        return f"{d['detected_within']}/{d['n']}"
    fa = f["false_alarms"]
    return f"FA {fa['max']} / {fa['rate']:.1e}"


def main(out_dir: Path, n_seeds: int = N_SEEDS, *, quiet: bool = False) -> int:
    from .provenance import header_line, provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    prov = provenance()
    results = {axis: run_axis(axis, n_seeds) for axis in AXES}
    text = render(results, n_seeds).replace("\n\n", "\n\n" + header_line(prov) + "\n\n", 1)
    (out_dir / "sweep.md").write_text(text, encoding="utf-8")
    (out_dir / "sweep.json").write_text(
        json.dumps({"n_seeds": n_seeds, "provenance": prov,
                    "axes": {a: {k: v for k, v in AXES[a].items() if k not in ("specs", "declared_specs")}
                             for a in AXES},
                    "results": {a: {str(p): r for p, r in pts.items()} for a, pts in results.items()}},
                   indent=2, default=str), encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path.cwd() / "results", int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS))
