"""Phase 1 experiment grid: 5 scenarios x 5 estimator variants x N seeds.

Every scenario declares the same constraint (closed boundary, m1 + m2 = 100 kg).
In three of them that constraint is true throughout; in the other two something
the constraint author did not know about makes it stale or misleading.

Each scenario is run over N_SEEDS independent seeds (simulation and degradation
seeds offset together). Tables report mean ± sd across seeds. A single seed is a
realization, not a result.

    python run_experiments.py            -> prints markdown tables, writes results/
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from set_lcm.degrade import DegradeConfig, observe
from set_lcm.evaluate import evaluate
from set_lcm.runner import EstimatorSpec, run
from set_lcm.schema import ConstraintSet
from set_lcm.simulator import SimConfig, simulate

SEED = 20260911
N_SEEDS = 20
SEED_STRIDE = 1000


@dataclass(frozen=True)
class Scenario:
    sim: SimConfig
    deg: DegradeConfig
    fault_onset: int | None
    windows: dict[str, tuple[int, int]]
    note: str


SCENARIOS: dict[str, Scenario] = {
    "closed_noise": Scenario(
        sim=SimConfig(seed=SEED),
        deg=DegradeConfig(seed=SEED + 1, dropout_p=0.05),
        fault_onset=None,
        windows={"steady": (300, 600)},
        note="Closed system, 2 kg noise, 5% random dropout. Constraint is TRUE.",
    ),
    "closed_blackout_pumpbias": Scenario(
        sim=SimConfig(seed=SEED, pump_actual=0.12),
        deg=DegradeConfig(seed=SEED + 2, dropout_p=0.05, blackout=(1, 100, 200)),
        fault_onset=50,
        windows={"blackout": (100, 200), "recovery": (200, 260), "steady": (300, 600)},
        note="Closed system; sensor 2 dark for steps 100-200 while the pump (on from step 50) "
             "actually delivers 0.12 kg/s against a commanded 0.10. Constraint is TRUE; the MODEL "
             "is wrong from step 50, so flags after 50 are detections of a parameter error.",
    ),
    "closed_blackout_noisy_valve": Scenario(
        sim=SimConfig(seed=SEED, transfer_noise_std=0.10),
        deg=DegradeConfig(seed=SEED + 5, dropout_p=0.05, blackout=(1, 100, 400)),
        fault_onset=0,
        windows={"blackout": (100, 400), "recovery": (400, 460), "steady": (460, 600)},
        note="Closed system; sensor 2 dark for steps 100-400 while an unmodeled valve moves mass "
             "between reservoirs at random (σ = 0.10 kg/s per step, zero mean; 4x the filter's Q). "
             "Pure observability loss, no parameter bias. Constraint is TRUE; the filter's noise "
             "model is wrong from step 0, so there is no false-alarm window -- 'detect' shows how "
             "many seeds ever flag the miscalibration.",
    ),
    "leak_stale_constraint": Scenario(
        sim=SimConfig(seed=SEED, leak_rate=0.05),
        deg=DegradeConfig(seed=SEED + 3, dropout_p=0.05),
        fault_onset=300,
        windows={"pre_leak": (0, 300), "leak_and_after": (300, 600)},
        note="10 kg leaks from reservoir 2 during steps 300-500. The declared closed-boundary "
             "constraint becomes STALE at step 300.",
    ),
    "bias_quant_delay": Scenario(
        sim=SimConfig(seed=SEED),
        deg=DegradeConfig(seed=SEED + 4, dropout_p=0.05, bias=(0, 3.0, 200), quant_step=0.5, delay_steps=5),
        fault_onset=200,
        windows={"pre_bias": (0, 200), "post_bias": (200, 600)},
        note="Closed system; sensor 1 acquires an undeclared +3 kg bias at step 200; 0.5 kg "
             "quantization; 5-step arrival delay. Constraint is TRUE but the evidence is not.",
    ),
}

SPECS = [
    EstimatorSpec("hold_last", "hold_last", None),
    EstimatorSpec("kf", "kf", None),
    EstimatorSpec("kf+soft(1/lam=4)", "kf", "soft", lam=0.25),
    EstimatorSpec("kf+hard", "kf", "hard"),
    EstimatorSpec("kf+hard+guard", "kf", "hard", guard=True),
]


def constraint_for(truth) -> ConstraintSet:
    return ConstraintSet(
        version="closed-boundary-v1",
        A=np.array([[1.0, 1.0]]),
        b=np.array([truth.total0]),
        description=f"m1 + m2 = {truth.total0:g} kg (boundary assumed closed)",
    )


def _ms(vals) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def aggregate(per_seed: list[dict]) -> dict:
    """per_seed: list over seeds of {estimator: metrics}. Returns {estimator: aggregated}."""
    out = {}
    for est in per_seed[0]:
        ms = [ps[est] for ps in per_seed]
        a = {k: _ms([m[k] for m in ms]) for k in (
            "rmse_all", "coverage95", "mean_abs_res_pre", "mean_abs_res_post",
            "mean_correction_norm", "latency_us_p50", "latency_us_p99")}
        a["rmse_by_window"] = {w: _ms([m["rmse_by_window"][w] for m in ms]) for w in ms[0]["rmse_by_window"]}
        a["coverage95_by_window"] = {w: _ms([m["coverage95_by_window"][w] for m in ms]) for w in ms[0]["coverage95_by_window"]}
        fa = [m["false_alarms"] for m in ms]
        a["false_alarms"] = {"mean": float(np.mean(fa)), "max": int(max(fa)),
                             "rate": float(np.mean([m["fa_rate"] for m in ms]))}
        dd = [m["detection_delay_steps"] for m in ms]
        det = [d for d in dd if d is not None]
        a["detection_delay_steps"] = {"mean": float(np.mean(det)) if det else None, "missed": len(dd) - len(det)}
        a["held_steps"] = _ms([m["status_counts"].get("model_inconsistent", 0) for m in ms])
        a["solver_failures"] = int(sum(m["solver_failures"] for m in ms))
        a["n_seeds"] = len(ms)
        out[est] = a
    return out


def run_scenario(name: str, n_seeds: int = N_SEEDS, specs=SPECS) -> tuple[dict, list[dict], dict]:
    sc = SCENARIOS[name]
    per_seed: list[dict] = []
    for i in range(n_seeds):
        sim = replace(sc.sim, seed=sc.sim.seed + SEED_STRIDE * i)
        deg = replace(sc.deg, seed=sc.deg.seed + SEED_STRIDE * i)
        truth = simulate(sim)
        obs = observe(truth, deg)
        cs = constraint_for(truth)
        per_seed.append({
            spec.name: evaluate(run(truth, obs, cs, spec, sim.m0, deg.delay_steps), truth, sc.fault_onset, sc.windows)
            for spec in specs
        })
    meta = {"sim": asdict(sc.sim), "deg": asdict(sc.deg), "constraint": "closed-boundary-v1",
            "fault_onset": sc.fault_onset, "windows": sc.windows, "note": sc.note,
            "n_seeds": n_seeds, "seed_stride": SEED_STRIDE}
    return aggregate(per_seed), per_seed, meta


def fms(d: dict | None, nd: int = 2) -> str:
    return "—" if d is None else f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def table(name: str, sc: Scenario, agg: dict) -> str:
    win = list(sc.windows)
    head = ["estimator", "RMSE all"] + [f"RMSE {w}" for w in win] + \
           ["cov95", "|res| post", "|corr|", "FA (max / rate)", "detect (mean / missed)", "held steps", "lat p50 µs"]
    rows = [head, ["---"] * len(head)]
    for est, a in agg.items():
        dd = a["detection_delay_steps"]
        det = "—" if sc.fault_onset is None else (f"{dd['mean']:.0f} / {dd['missed']}" if dd["mean"] is not None else f"— / {dd['missed']}")
        rows.append([
            est, fms(a["rmse_all"]),
            *[fms(a["rmse_by_window"][w]) for w in win],
            fms(a["coverage95"]),
            "—" if a["mean_abs_res_post"] is None else f"{a['mean_abs_res_post']['mean']:.1e}",
            fms(a["mean_correction_norm"]),
            f"{a['false_alarms']['max']} / {a['false_alarms']['rate']:.1e}", det,
            f"{a['held_steps']['mean']:.0f}",
            f"{a['latency_us_p50']['mean']:.0f}",
        ])
    return "\n".join([f"### {name}", "", sc.note, "", *("| " + " | ".join(r) + " |" for r in rows), ""])


def main() -> int:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    summary = {"seed": SEED, "n_seeds": N_SEEDS, "scenarios": {}}
    md = ["# SET + LCM Phase 1 results", "",
          f"Two-reservoir material transfer, 600 steps, hidden truth, {N_SEEDS} seeds per scenario "
          "(mean ± sd across seeds). Declared constraint: m1 + m2 = 100 kg.",
          "RMSE in kg over both reservoirs. cov95 = fraction of steps where truth lies inside the reported "
          "±1.96σ interval. |res| post = mean |A x − b| after projection. A flag rejects the joint hypothesis "
          "(constraint ∧ model ∧ calibrated uncertainty); onset = first step at which that hypothesis is false. "
          "FA = worst-seed count / mean per-step rate of flags before onset (whole run if no onset). "
          "detect = mean steps from onset to first flag / seeds that never flagged. held = mean steps the "
          "guard reported model_inconsistent. Flags use χ²₁(0.999) with a 3-step debounce.", ""]
    for name, sc in SCENARIOS.items():
        agg, per_seed, meta = run_scenario(name)
        summary["scenarios"][name] = {"meta": meta, "aggregate": agg, "per_seed": per_seed}
        md.append(table(name, sc, agg))
        print(md[-1], flush=True)
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
