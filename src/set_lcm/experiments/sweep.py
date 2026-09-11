"""Fault-magnitude sweep.

Where does the guard's dead band start, at what fault size does hard projection
begin to hurt, and what does a principled soft mode buy when the declared total
is itself uncertain?

Axes (each point: 20 seeds, same SEED / SEED_STRIDE as the Phase 1 grid):
  declared_total_error   the declared total b is off by delta kg (closed_noise otherwise)
  sensor_bias            undeclared sensor-1 bias of b kg from step 200 (bias_quant_delay otherwise)
  leak_rate              leak of r kg/s from reservoir 2 over steps 300-500
  uncertain_total        b = total0 + N(0, sigma_b^2) drawn per seed; adds kf+soft(lam = 1/sigma_b^2),
                         the pseudo-measurement whose variance IS the declared uncertainty

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
from ..testbed.runner import EstimatorSpec, run
from ..testbed.simulator import simulate
from .phase1 import DETECT_WITHIN, N_SEEDS, SCENARIOS, SEED, SEED_STRIDE, Scenario, aggregate, declared_prior

KF = EstimatorSpec("kf", "kf", None)
HARD = EstimatorSpec("kf+hard", "kf", "hard")
GUARD = EstimatorSpec("kf+hard+guard", "kf", "hard", guard=True)
SOFT_NAME = "kf+soft(lam=1/sigma_b^2)"

AXES: dict[str, dict] = {
    "declared_total_error": {
        "base": "closed_noise", "points": (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0), "unit": "kg",
        "window": "steady", "specs": (KF, HARD, GUARD),
        "note": "The declared total is wrong by delta. The joint hypothesis is false from step 0 for delta > 0.",
    },
    "sensor_bias": {
        "base": "bias_quant_delay", "points": (0.5, 1.0, 2.0, 3.0), "unit": "kg",
        "window": "post_bias", "specs": (KF, HARD, GUARD),
        "note": "Undeclared sensor-1 bias from step 200, with 0.5 kg quantization and 5-step delay.",
    },
    "leak_rate": {
        "base": "leak_stale_constraint", "points": (0.005, 0.01, 0.02, 0.05), "unit": "kg/s",
        "window": "leak_and_after", "specs": (KF, HARD, GUARD),
        "note": "Leak from reservoir 2 over steps 300-500; total lost = 200 x rate.",
    },
    "uncertain_total": {
        "base": "closed_noise", "points": (0.1, 0.3, 0.5, 1.0, 2.0), "unit": "kg (sigma_b)",
        "window": "steady", "specs": (KF, HARD, GUARD),   # + soft with lam = 1/sigma_b^2, added per point
        "note": "b = total0 + N(0, sigma_b^2) per seed. The guard tests the constraint as if exact; "
                "the soft variant uses the declared uncertainty as the pseudo-measurement variance.",
    },
}


def _point_config(axis: str, value: float, i: int, base: Scenario):
    """Return (sim, deg, cs_fn, onset, specs) for one sweep point and seed index."""
    sim = replace(base.sim, seed=base.sim.seed + SEED_STRIDE * i)
    deg = replace(base.deg, seed=base.deg.seed + SEED_STRIDE * i)
    specs = list(AXES[axis]["specs"])
    if axis == "declared_total_error":
        onset = None if value == 0.0 else 0
        cs_fn = lambda truth: _cs(truth.total0 + value)
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
        cs_fn = lambda truth: _cs(truth.total0 + offset, b_std=value)
        specs.append(EstimatorSpec(SOFT_NAME, "kf", "soft", lam=1.0 / value ** 2))
    else:
        raise KeyError(axis)
    return sim, deg, cs_fn, onset, specs


def _cs(total: float, b_std: float | None = None) -> ConstraintSet:
    desc = f"m1 + m2 = {total:.4g} kg" + (f" (declared std {b_std:g} kg)" if b_std else "")
    return ConstraintSet("closed-boundary-sweep", np.array([[1.0, 1.0]]), np.array([total]), desc)


def run_point(axis: str, value: float, n_seeds: int = N_SEEDS) -> dict:
    base = SCENARIOS[AXES[axis]["base"]]
    per_seed = []
    for i in range(n_seeds):
        sim, deg, cs_fn, onset, specs = _point_config(axis, value, i, base)
        truth = simulate(sim)
        obs = observe(truth, deg)
        cs = cs_fn(truth)
        m0, m0_std = declared_prior(base, sim)
        per_seed.append({
            spec.name: evaluate(run(truth, obs, cs, spec, m0, m0_std), truth, onset, base.windows, cs=cs)
            for spec in specs
        })
    return aggregate(per_seed)


def run_axis(axis: str, n_seeds: int = N_SEEDS, points=None) -> dict[float, dict]:
    pts = AXES[axis]["points"] if points is None else tuple(points)
    return {float(p): run_point(axis, float(p), n_seeds) for p in pts}


def _cell(a: dict, window: str) -> str:
    return f"{a['rmse_by_window'][window]['mean']:.2f} / {a['coverage95_by_window'][window]['mean']:.2f} / {a['nz_rms_by_window'][window]['mean']:.2f}"


def render(results: dict[str, dict[float, dict]], n_seeds: int) -> str:
    md = ["# Fault-magnitude sweep", "",
          f"{n_seeds} seeds per point. Cells are (RMSE kg / cov95 / nz) over the fault window; "
          f"det = seeds flagged within {DETECT_WITHIN} steps of onset; held = mean steps the guard reported "
          "model_inconsistent. nz = RMS normalised error (1.0 calibrated, >1 over-confident).", ""]
    for axis, pts in results.items():
        cfg = AXES[axis]
        w = cfg["window"]
        specs = list(next(iter(pts.values())).keys())
        md += [f"## {axis}", "", cfg["note"], "",
               "| " + f"{axis} [{cfg['unit']}]" + " | " + " | ".join(f"{s} (RMSE/cov/nz)" for s in specs) +
               " | " + " | ".join(f"{s} det" for s in specs if "guard" in s or "soft" in s) + " | guard held |",
               "| " + " | ".join(["---"] * (1 + len(specs) + sum(1 for s in specs if "guard" in s or "soft" in s) + 1)) + " |"]
        for p, agg in pts.items():
            cells = [f"{p:g}"] + [_cell(agg[s], w) for s in specs]
            dets = []
            for s in specs:
                if "guard" in s or "soft" in s:
                    d = agg[s]["detection"]
                    dets.append(f"{d['detected_within']}/{d['n']}" if d["applicable"] else "—")
            held = f"{agg['kf+hard+guard']['held_steps']['mean']:.0f}" if "kf+hard+guard" in agg else "—"
            md.append("| " + " | ".join(cells + dets + [held]) + " |")
        md.append("")
    return "\n".join(md)


def main(out_dir: Path, n_seeds: int = N_SEEDS, *, quiet: bool = False) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    results = {axis: run_axis(axis, n_seeds) for axis in AXES}
    text = render(results, n_seeds)
    (out_dir / "sweep.md").write_text(text, encoding="utf-8")
    (out_dir / "sweep.json").write_text(
        json.dumps({"n_seeds": n_seeds, "axes": {a: {k: v for k, v in AXES[a].items() if k != "specs"} for a in AXES},
                    "results": {a: {str(p): r for p, r in pts.items()} for a, pts in results.items()}},
                   indent=2, default=str), encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path.cwd() / "results", int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS))
