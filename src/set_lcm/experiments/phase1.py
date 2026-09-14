"""Experiment grid: 7 scenarios x 11 estimator specs (SPECS) x N seeds.

Each scenario builds its constraint with its own builder (Scenario.constraint_builder).
Six of them declare the same exact constraint (closed boundary, m1 + m2 = 100 kg, no
uncertainty on b): in four that constraint is true throughout; in the other two
something the constraint author did not know about makes it stale or misleading.
The seventh, closed_uncertain_total, declares a total that is itself uncertain --
b = 100 kg + an offset drawn per seed from N(0, 1 kg^2) -- together with that
uncertainty (b_var = 1 kg^2), so the joint hypothesis holds; every projecting spec
honours b_var because it is on the ConstraintSet.

Each scenario is run over N_SEEDS independent seeds (simulation and degradation
seeds offset together). Tables report mean ± sd across seeds. A single seed is a
realization, not a result.

Hidden truth is read here and routed, and nowhere on the estimator side: the
runner is handed PublicInputs.from_truth(truth) -- the step clock and the
commanded input -- and, for the oracle bound only, the hidden (u_actual, leak)
through run()'s explicit oracle_inputs keyword (oracle_inputs_for / run_spec).
The evaluator reads the truth afterwards, to score.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..schema import ConstraintSet
from ..testbed.degrade import DegradeConfig, observe
from ..testbed.evaluate import evaluate
from ..testbed.inputs import PublicInputs
from ..testbed.runner import EstimatorSpec, RunResult, run
from ..testbed.simulator import SimConfig, Truth, simulate

SEED = 20260911
N_SEEDS = 20
SEED_STRIDE = 1000
DETECT_WITHIN = 100   # steps after onset within which a flag counts as a timely detection


def constraint_for(truth) -> ConstraintSet:
    return ConstraintSet(
        version="closed-boundary-v1",
        A=np.array([[1.0, 1.0]]),
        b=np.array([truth.total0]),
        description=f"m1 + m2 = {truth.total0:g} kg (boundary assumed closed)",
        row_units=("kg",),
    )


@dataclass(frozen=True)
class ExactTotal:
    """Constraint builder: m1 + m2 = total0, declared exact (b_var None). What every
    scenario declared before b_var existed; the seed index is not used."""

    def __call__(self, truth: Truth, i: int) -> ConstraintSet:
        return constraint_for(truth)

    def describe(self) -> dict:
        return {"kind": "ExactTotal"}


@dataclass(frozen=True)
class UncertainTotal:
    """Constraint builder: m1 + m2 = total0 + offset, offset ~ N(0, std^2) drawn for seed
    index i from default_rng(seed + SEED_STRIDE * i), declared WITH its uncertainty
    (b_var = [std^2]). The declared b is off, by exactly as much as it says it may be."""
    std: float
    seed: int

    def offset(self, i: int) -> float:
        return float(np.random.default_rng(self.seed + SEED_STRIDE * i).normal(0.0, self.std))

    def __call__(self, truth: Truth, i: int) -> ConstraintSet:
        total = truth.total0 + self.offset(i)
        return ConstraintSet(
            version="closed-boundary-uncertain-v1",
            A=np.array([[1.0, 1.0]]),
            b=np.array([total]),
            description=f"m1 + m2 = {total:.4g} kg, declared b_var = {self.std ** 2:g} kg^2 "
                        f"(boundary assumed closed)",
            b_var=np.array([self.std ** 2]),
            row_units=("kg",),
        )

    def describe(self) -> dict:
        return {"kind": "UncertainTotal", "std": self.std, "b_var": self.std ** 2, "seed": self.seed}


@dataclass(frozen=True)
class Scenario:
    sim: SimConfig
    deg: DegradeConfig
    fault_onset: int | None
    windows: dict[str, tuple[int, int]]
    note: str
    # What the estimator is told about the initial state. None means "the declared
    # initial fill equals the simulator's" -- a scenario choice, stated here, not a leak.
    declared_m0: tuple[float, float] | None = None
    declared_m0_std: float = 5.0
    # Direction in state space along which the scenario's fault pushes the estimate;
    # the evaluator reports d(f) = f^T A^T (A P A^T + Sigma_b)^-1 A f for it (Sigma_b = 0 if exact).
    fault_direction: tuple[float, float] | None = None
    # The per-scenario constraint builder: (truth, seed index) -> the declared ConstraintSet.
    constraint_builder: ExactTotal | UncertainTotal = ExactTotal()


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
        fault_direction=(-1.0, 1.0),
        note="Closed system; sensor 2 dark for steps 100-200 while the pump (on from step 50) "
             "actually delivers 0.12 kg/s against a commanded 0.10. Constraint is TRUE; the MODEL "
             "is wrong from step 50. The fault moves mass along (-1, +1), which lies in null(A).",
    ),
    "closed_blackout_noisy_valve": Scenario(
        sim=SimConfig(seed=SEED, transfer_noise_std=0.10),
        deg=DegradeConfig(seed=SEED + 5, dropout_p=0.05, blackout=(1, 100, 400)),
        fault_onset=0,
        windows={"blackout": (100, 400), "recovery": (400, 460), "steady": (460, 600)},
        fault_direction=(-1.0, 1.0),
        note="Closed system; sensor 2 dark for steps 100-400 while an unmodeled valve moves mass "
             "between reservoirs at random (σ = 0.10 kg/s per step, zero mean; 4x the filter's Q). "
             "Pure observability loss, no parameter bias. Constraint is TRUE; the filter's noise "
             "model is wrong from step 0, so there is no false-alarm window. The disturbance acts "
             "along (-1, +1), which lies in null(A).",
    ),
    "leak_stale_constraint": Scenario(
        sim=SimConfig(seed=SEED, leak_rate=0.05),
        deg=DegradeConfig(seed=SEED + 3, dropout_p=0.05),
        fault_onset=300,
        windows={"pre_leak": (0, 300), "leak_and_after": (300, 600)},
        fault_direction=(0.0, -1.0),
        note="10 kg leaks from reservoir 2 during steps 300-500. The declared closed-boundary "
             "constraint becomes STALE at step 300. The fault direction (0, -1) has a component "
             "along row(A).",
    ),
    "bias_quant_delay": Scenario(
        sim=SimConfig(seed=SEED),
        deg=DegradeConfig(seed=SEED + 4, dropout_p=0.05, bias=(0, 3.0, 200), quant_step=0.5, delay_steps=5),
        fault_onset=200,
        windows={"pre_bias": (0, 200), "post_bias": (200, 600)},
        fault_direction=(1.0, 0.0),
        note="Closed system; sensor 1 acquires an undeclared +3 kg bias at step 200; 0.5 kg "
             "quantization; 5-step arrival delay. Constraint is TRUE but the evidence is not. "
             "The fault direction (1, 0) has a component along row(A).",
    ),
    "closed_wrong_prior": Scenario(
        sim=SimConfig(seed=SEED),
        deg=DegradeConfig(seed=SEED + 6, dropout_p=0.05),
        fault_onset=None,
        windows={"settle": (0, 50), "steady": (300, 600)},
        declared_m0=(74.0, 26.0), declared_m0_std=5.0,
        note="Closed system as closed_noise, but the estimator is initialised from a DECLARED "
             "initial fill of 74/26 kg (truth 70/30) with 5 kg prior std. Constraint is TRUE; "
             "the prior is wrong and has to be forgotten from the evidence.",
    ),
    "closed_uncertain_total": Scenario(
        sim=SimConfig(seed=SEED),
        deg=DegradeConfig(seed=SEED + 8, dropout_p=0.05),
        fault_onset=None,
        windows={"steady": (300, 600)},
        constraint_builder=UncertainTotal(std=1.0, seed=SEED + 9),
        note="Closed system as closed_noise (own degradation seed). The declared total is "
             "100 kg + an offset drawn per seed from N(0, 1 kg²), and the constraint DECLARES "
             "that uncertainty (b_var = 1 kg²): b is within its declared uncertainty, so the "
             "joint hypothesis holds and every flag is a false alarm. Every projecting spec "
             "honours b_var: hard is the Kalman update with pseudo-measurement variance 1 kg², "
             "soft adds its 1/λ on top.",
    ),
}

SPECS = [
    EstimatorSpec("hold_last", "hold_last", None),
    EstimatorSpec("kf", "kf", None),
    EstimatorSpec("kf+soft(1/lam=4)", "kf", "soft", lam=0.25),
    EstimatorSpec("kf+hard", "kf", "hard"),
    EstimatorSpec("kf+hard+guard", "kf", "hard", guard=True),
    # fed-back projection: the projected (x*, P*) becomes the filter's state
    EstimatorSpec("kf+hard+fb", "kf", "hard", feedback=True),
    EstimatorSpec("kf+hard+fb+guard", "kf", "hard", guard=True, feedback=True),
    # closure in the noise model instead of in a constraint row (mode None: consistency stat
    # computed and flagged on its marginal, never projected); and the same with a row
    EstimatorSpec("kf_closedq", "kf_closedq", None),
    EstimatorSpec("kf_closedq+hard", "kf_closedq", "hard"),
    # mode None: the consistency stat is still computed on the mass marginal and its flag
    # recorded, but nothing is ever projected; alpha and L are outputs with their own flags
    EstimatorSpec("kf_aug", "kf_aug", None),
    # BOUND, not a candidate: knows the hidden actual pump rate and leak (see runner.run)
    EstimatorSpec("oracle (bound)", "oracle", None),
]


def oracle_inputs_for(spec: EstimatorSpec, truth: Truth) -> tuple[np.ndarray, np.ndarray] | None:
    """THE ONE PLACE hidden truth is routed to the estimator side: the oracle (bound)
    gets the hidden actual pump parameter rate and the hidden leak as known inputs.
    Every other kind gets None, and run() refuses hidden inputs for any kind but
    "oracle". Truth.m is not routed anywhere."""
    if spec.kind != "oracle":
        return None
    return truth.u_actual, truth.leak


def run_spec(truth: Truth, obs, cs: ConstraintSet | None, spec: EstimatorSpec,
             m0: tuple[float, ...], m0_std: float) -> RunResult:
    """Run one spec on a simulated record. The runner sees the public inputs only;
    the oracle's hidden arrays go through oracle_inputs_for, explicitly."""
    return run(PublicInputs.from_truth(truth), obs, cs, spec, m0, m0_std,
               oracle_inputs=oracle_inputs_for(spec, truth))


def scenario_constraint(sc: Scenario, truth: Truth, i: int) -> ConstraintSet:
    """The ConstraintSet scenario sc declares for seed index i."""
    return sc.constraint_builder(truth, i)


def declared_prior(sc: Scenario, sim: SimConfig) -> tuple[tuple[float, float], float]:
    """What the estimator is told about the initial state for this scenario."""
    m0 = sc.declared_m0 if sc.declared_m0 is not None else sim.m0
    return m0, sc.declared_m0_std


def _ms(vals) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def _detection(ms: list[dict]) -> dict:
    """Censored summary of the constraint flag's per-seed detection delays."""
    return _detection_summary([m["detection_delay_steps"] for m in ms], ms[0]["detection_censor_steps"])


def _detection_summary(delays: list[int | None], censor: int | None) -> dict:
    """Censored summary of per-seed detection delays. Never a mean over survivors."""
    n = len(delays)
    if censor is None:
        return {"n": n, "detected_any": 0, "within": DETECT_WITHIN, "detected_within": 0,
                "median_delay": None, "median_censored": None, "censor_at": None, "applicable": False}
    detected = [d for d in delays if d is not None]
    values = np.array([d if d is not None else censor for d in delays], dtype=float)
    med = float(np.median(values))
    return {
        "n": n,
        "detected_any": len(detected),
        "within": DETECT_WITHIN,
        "detected_within": int(sum(d <= DETECT_WITHIN for d in detected)),
        "median_delay": med,
        "median_censored": bool(med >= censor),
        "censor_at": int(censor),
        "applicable": True,
    }


def aggregate(per_seed: list[dict]) -> dict:
    """per_seed: list over seeds of {estimator: metrics}. Returns {estimator: aggregated}."""
    out = {}
    for est in per_seed[0]:
        ms = [ps[est] for ps in per_seed]
        a = {k: _ms([m[k] for m in ms]) for k in (
            "rmse_all", "coverage95", "nz_rms_all", "mean_abs_res_pre", "mean_abs_res_post",
            "mean_correction_norm", "latency_us_p50", "latency_us_p99")}
        windows = list(ms[0]["rmse_by_window"])
        for key in ("rmse_by_window", "coverage95_by_window", "nz_rms_by_window", "nz_rms_unproj_by_window"):
            a[key] = {w: _ms([m[key][w] for m in ms]) for w in windows}
        a["innov_z_mean_by_window"] = {
            w: [_ms([m["innov_z_mean_by_window"][w][i] for m in ms]) for i in range(len(ms[0]["innov_z_mean_by_window"][w]))]
            for w in windows}
        for key in ("rms_err_by_direction", "rms_err_unproj_by_direction"):
            if key in ms[0]:
                a[key] = {w: {d: _ms([m[key][w][d] for m in ms]) for d in ("row", "null")} for w in windows}
        if "detectability" in ms[0]:
            a["detectability"] = {**_ms([m["detectability"]["value"] for m in ms]),
                                  "window": ms[0]["detectability"]["window"],
                                  "direction": ms[0]["detectability"]["direction"]}
        fa = [m["false_alarms"] for m in ms]
        a["false_alarms"] = {"mean": float(np.mean(fa)), "max": int(max(fa)),
                             "rate": float(np.mean([m["fa_rate"] for m in ms]))}
        a["detection"] = _detection(ms)
        # evidence-side channel, summarised per sensor exactly like the constraint flag
        if all(m["cusum_applicable"] for m in ms):
            a["cusum"] = {}
            for i in range(len(ms[0]["cusum_false_alarms"])):
                fa_i = [m["cusum_false_alarms"][i] for m in ms]
                a["cusum"][f"s{i + 1}"] = {
                    "false_alarms": {"mean": float(np.mean(fa_i)), "max": int(max(fa_i)),
                                     "rate": float(np.mean([m["cusum_fa_rate"][i] for m in ms]))},
                    "detection": _detection_summary([m["cusum_detection_delay_steps"][i] for m in ms],
                                                    ms[0]["detection_censor_steps"]),
                }
        else:
            a["cusum"] = None
        # augmented-state outputs: parameter errors and one flag summary per extra component
        if all(m.get("aug") is not None for m in ms):
            augs = [m["aug"] for m in ms]
            a["aug"] = {
                "alpha_rmse_pump_on": _ms([g["alpha_rmse_pump_on"] for g in augs]),
                "alpha_rmse_pump_on_by_window": {w: _ms([g["alpha_rmse_pump_on_by_window"][w] for g in augs]) for w in windows},
                "L_rmse": _ms([g["L_rmse"] for g in augs]),
                "L_rmse_by_window": {w: _ms([g["L_rmse_by_window"][w] for g in augs]) for w in windows},
                "alpha_hat_end": _ms([g["alpha_hat_end"] for g in augs]),
                "alpha_sd_end": _ms([g["alpha_sd_end"] for g in augs]),
                "L_hat_end": _ms([g["L_hat_end"] for g in augs]),
                "L_sd_end": _ms([g["L_sd_end"] for g in augs]),
                "flags": {},
            }
            for name in augs[0]["flags"]:
                fa_i = [g["flags"][name]["false_alarms"] for g in augs]
                a["aug"]["flags"][name] = {
                    "false_alarms": {"mean": float(np.mean(fa_i)), "max": int(max(fa_i)),
                                     "rate": float(np.mean([g["flags"][name]["fa_rate"] for g in augs]))},
                    "detection": _detection_summary([g["flags"][name]["detection_delay_steps"] for g in augs],
                                                    ms[0]["detection_censor_steps"]),
                }
        else:
            a["aug"] = None
        a["held_steps"] = _ms([m["status_counts"].get("model_inconsistent", 0) for m in ms])
        a["solver_failures"] = int(sum(m["solver_failures"] for m in ms))
        a["n_seeds"] = len(ms)
        out[est] = a
    return out


def run_scenario(name: str, n_seeds: int = N_SEEDS, specs=SPECS) -> tuple[dict, list[dict], dict]:
    sc = SCENARIOS[name]
    per_seed: list[dict] = []
    declared: list[ConstraintSet] = []
    for i in range(n_seeds):
        sim = replace(sc.sim, seed=sc.sim.seed + SEED_STRIDE * i)
        deg = replace(sc.deg, seed=sc.deg.seed + SEED_STRIDE * i)
        truth = simulate(sim)
        obs = observe(truth, deg)
        cs = scenario_constraint(sc, truth, i)
        declared.append(cs)
        m0, m0_std = declared_prior(sc, sim)
        per_seed.append({
            spec.name: evaluate(run_spec(truth, obs, cs, spec, m0, m0_std), truth, sc.fault_onset, sc.windows,
                                cs=cs, fault_direction=sc.fault_direction)
            for spec in specs
        })
    meta = {"sim": asdict(sc.sim), "deg": asdict(sc.deg), "constraint": declared[0].version,
            "fault_onset": sc.fault_onset, "windows": sc.windows, "note": sc.note,
            "declared_m0": sc.declared_m0, "declared_m0_std": sc.declared_m0_std,
            "fault_direction": sc.fault_direction,
            "n_seeds": n_seeds, "seed_stride": SEED_STRIDE,
            "constraint_builder": sc.constraint_builder.describe(),
            "declared_b_per_seed": [float(cs.b[0]) for cs in declared],
            "declared_b_var": None if declared[0].b_var is None else declared[0].b_var.tolist()}
    return aggregate(per_seed), per_seed, meta


def iter_runs(name: str, spec: EstimatorSpec, n_seeds: int = N_SEEDS):
    """Yield (seed_index, truth, obs, cs, RunResult) for one scenario and one estimator spec.
    Used by the calibration and sweep modules, which need the raw per-step record."""
    sc = SCENARIOS[name]
    for i in range(n_seeds):
        sim = replace(sc.sim, seed=sc.sim.seed + SEED_STRIDE * i)
        deg = replace(sc.deg, seed=sc.deg.seed + SEED_STRIDE * i)
        truth = simulate(sim)
        obs = observe(truth, deg)
        cs = scenario_constraint(sc, truth, i)
        m0, m0_std = declared_prior(sc, sim)
        yield i, truth, obs, cs, run_spec(truth, obs, cs, spec, m0, m0_std)


def fms(d: dict | None, nd: int = 2) -> str:
    return "—" if d is None else f"{d['mean']:.{nd}f} ± {d['sd']:.{nd}f}"


def fm(d: dict | None, nd: int = 2) -> str:
    return "—" if d is None else f"{d['mean']:.{nd}f}"


def _row(rows_: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(r) + " |" for r in rows_]


def _det_cells(det: dict) -> tuple[str, str]:
    """(k/n detected within DETECT_WITHIN, censored median delay) as table cells."""
    if not det["applicable"]:
        return "—", "—"
    within = f"{det['detected_within']}/{det['n']}"
    med = f"> {det['censor_at']}" if det["median_censored"] else f"{det['median_delay']:.0f}"
    return within, med


def table(name: str, sc: Scenario, agg: dict) -> str:
    """Two tables per scenario: accuracy/calibration per window, then reconciliation/detection;
    a third, for estimators that carry augmented state, with the parameter outputs."""
    win = list(sc.windows)

    head1 = ["estimator"]
    for w in win:
        head1 += [f"RMSE {w}", f"cov95 {w}", f"nz {w}", f"err row/null {w}", f"z̄ s1/s2 {w}"]
    rows1 = [head1, ["---"] * len(head1)]
    for est, a in agg.items():
        r = [est]
        for w in win:
            d = a.get("rms_err_by_direction", {}).get(w)
            z = a["innov_z_mean_by_window"][w]
            r += [fms(a["rmse_by_window"][w]), fm(a["coverage95_by_window"][w]), fm(a["nz_rms_by_window"][w]),
                  "—" if d is None else f"{fm(d['row'])} / {fm(d['null'])}",
                  " / ".join("—" if zi is None else f"{zi['mean']:+.2f}" for zi in z)]
        rows1.append(r)

    head2 = ["estimator", "RMSE all", "cov95 all", "nz all", "|res| post", "|corr|", "FA (max / rate)",
             f"detected ≤{DETECT_WITHIN} (k/n)", "median delay", "held steps", "d(f)",
             "CUSUM s1 (k/n, median)", "CUSUM s2 (k/n, median)", "CUSUM FA max (s1 / s2)",
             "lat p50 µs (this machine)"]
    rows2 = [head2, ["---"] * len(head2)]
    for est, a in agg.items():
        within, med = _det_cells(a["detection"])
        dfa = a.get("detectability")
        cus = a.get("cusum")
        if cus is None:
            cusum_cells = ["—", "—", "—"]
        else:
            cusum_cells = [", ".join(_det_cells(cus[s]["detection"])) if cus[s]["detection"]["applicable"] else "—"
                           for s in ("s1", "s2")]
            cusum_cells.append(f"{cus['s1']['false_alarms']['max']} / {cus['s2']['false_alarms']['max']}")
        rows2.append([
            est, fms(a["rmse_all"]), fm(a["coverage95"]), fm(a["nz_rms_all"]),
            "—" if a["mean_abs_res_post"] is None else f"{a['mean_abs_res_post']['mean']:.1e}",
            fms(a["mean_correction_norm"]),
            f"{a['false_alarms']['max']} / {a['false_alarms']['rate']:.1e}",
            within, med, f"{a['held_steps']['mean']:.0f}",
            "—" if dfa is None else f"{dfa['mean']:.3g}",
            *cusum_cells,
            f"{a['latency_us_p50']['mean']:.0f}",
        ])

    parts = [f"### {name}", "", sc.note, "",
             "Accuracy and calibration per window:", "", *_row(rows1), "",
             "Reconciliation and detection:", "", *_row(rows2), ""]

    aug = {est: a["aug"] for est, a in agg.items() if a.get("aug") is not None}
    if aug:
        per_w = " / ".join(win)
        head3 = ["estimator", "alpha RMSE (pump on)", f"alpha RMSE (pump on) {per_w}", "L RMSE", f"L RMSE {per_w}",
                 "alpha flag (k/n, median)", "L flag (k/n, median)", "FA max (alpha / L)", "σ_α / σ_L at run end"]
        rows3 = [head3, ["---"] * len(head3)]
        for est, g in aug.items():
            fl = g["flags"]
            rows3.append([
                est, fms(g["alpha_rmse_pump_on"], 3),
                " / ".join(fm(g["alpha_rmse_pump_on_by_window"][w], 3) for w in win),
                fms(g["L_rmse"], 4),
                " / ".join(fm(g["L_rmse_by_window"][w], 4) for w in win),
                ", ".join(_det_cells(fl["alpha"]["detection"])),
                ", ".join(_det_cells(fl["L"]["detection"])),
                f"{fl['alpha']['false_alarms']['max']} / {fl['L']['false_alarms']['max']}",
                f"{fm(g['alpha_sd_end'], 3)} / {fm(g['L_sd_end'], 4)}",
            ])
        parts += ["Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):", "", *_row(rows3), ""]

    return "\n".join(parts)


LEGEND = (
    "RMSE in kg over both reservoirs. cov95 = fraction of steps where truth lies inside the reported "
    "±1.96σ interval. nz = RMS of the normalised error eᵢ/σᵢ (1.0 when calibrated; >1 over-confident). "
    "err row/null = RMS error of the reported state along row(A) and null(A) (for A = [1, 1]: the sum "
    "direction and the difference direction). z̄ s1/s2 = mean normalised innovation (y − H x_pred)/√Sᵢᵢ per "
    "sensor over the window's sampling steps (0 when the filter's prediction is unbiased; — where the sensor "
    "is dark or the estimator has no prediction). |res| post = mean |A x − b| after projection. "
    "A flag rejects the joint hypothesis (constraint ∧ model ∧ calibrated uncertainty); onset = first "
    "step at which that hypothesis is false. FA = worst-seed count / mean per-step rate of flags before "
    "onset (whole run if no onset). detected ≤N = seeds flagged within N steps of onset; median delay is "
    "the median over all seeds with never-flagged seeds censored at the end of the run (\"> T\" when the "
    "median itself is censored). held = mean steps the guard reported model_inconsistent. "
    "d(f) = fᵀAᵀ(APAᵀ)⁻¹Af for the scenario's fault direction on the unprojected P at the end of the "
    "first post-onset window; 0 means the consistency test is structurally blind to that fault. "
    "Flags use χ²(rank A)(0.999) with a 3-step debounce. "
    "Declared constraint uncertainty: where the constraint declares b_var (closed_uncertain_total only), "
    "the statistic is rᵀ(APAᵀ + Σ_b)⁻¹r and the joint hypothesis includes 'b is within its declared "
    "uncertainty'; hard projection is the Kalman update with pseudo-measurement variance Σ_b (residual "
    "post > 0), soft adds its 1/λ on top, and d(f) uses APAᵀ + Σ_b. "
    "CUSUM sᵢ = the evidence-side channel: a two-sided CUSUM (k = 0.5, h = 8) on sensor i's normalised "
    "innovation z = (y − H x_pred)/√Sᵢᵢ, updated when the observation is ingested and stamped at that "
    "report step, so its delay includes arrival delay; cells are (seeds alarmed within N steps of onset, "
    "censored median delay) and CUSUM FA max is the worst-seed count of pre-onset alarms per sensor. "
    "It reads no constraint; hold-last has no prediction, hence no innovation and no CUSUM. "
    "Augmented-state outputs (kf_aug only): the filter's state is [m1, m2, alpha, L] with alpha the pump "
    "scale (m1' = m1 − αu dt, m2' = m2 + αu dt − L dt; prior α ~ N(1, 0.1²), L ~ N(0, 0.02²), random walks "
    "1e-3 and 2e-3 kg/s per step); its RMSE / cov95 / nz rows above are the mass marginal, never projected. "
    "alpha RMSE (pump on) = RMS of α̂ − u_actual/u_commanded over the steps where the pump is commanded on "
    "(α is unobservable when u = 0; u_actual is the hidden parameter rate, so the per-step pump fluctuation "
    "counts as process noise, not as α error). L RMSE = RMS of L̂ − leak (kg/s) over the run / window. "
    "alpha flag / L flag = |α̂ − 1| / σ_α > 3.29 and |L̂| / σ_L > 3.29 (two-sided 0.001) for 3 consecutive "
    "reports; cells and FA max as for the constraint flag, with the same onset. σ_α / σ_L at run end = the "
    "filter's own reported sd of each parameter at the last step (mean over seeds). These flags name a "
    "parameter, not a cause: a sensor bias that the filter can only explain through the pump will raise the "
    "alpha flag. "
    "Baselines: kf_closedq is the kf with closure written into its process noise instead of into a constraint "
    "row, Q = σ_q² dt² B Bᵀ + ε I with B = (−1, 1), σ_q = 0.01 kg/s (the simulator's declared pump fluctuation) "
    "and ε = 1e-8; it has no constraint row and does not know b, and its consistency stat is computed on its "
    "own marginal but never projected (kf_closedq+hard adds the row). kf+hard+fb feeds the projected (x*, P*) "
    "back into the filter's state after every applied projection (under arrival delay, the projection of the "
    "filter's own state at its last ingested step); +guard stops feeding back while the flag holds. "
    "oracle (bound) is a KF given the HIDDEN actual pump parameter rate and the hidden leak as known inputs "
    "with Q = ε I only: a bound on what a perfect model of the inputs could do, never a candidate; it does not "
    "model the per-step pump fluctuation or the valve transfer, so it is over-confident wherever the truth is "
    "not deterministic given those inputs."
)


def main(out_dir: Path, *, quiet: bool = False) -> int:
    """Run the whole grid, print the markdown tables, write summary.md / summary.json to out_dir."""
    from .provenance import header_line, provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    prov = provenance()
    summary = {"schema_version": "fsre-phase1-summary-v1", "seed": SEED, "n_seeds": N_SEEDS,
               "detect_within": DETECT_WITHIN, "provenance": prov, "scenarios": {}}
    md = ["# SET + LCM Phase 1 results", "", header_line(prov), "",
          f"Two-reservoir material transfer, 600 steps, hidden truth, {N_SEEDS} seeds per scenario "
          "(mean ± sd across seeds where shown). Declared constraint: m1 + m2 = 100 kg, exact, in every "
          "scenario except closed_uncertain_total, which declares m1 + m2 = 100 kg + an offset drawn per "
          "seed from N(0, 1 kg²) together with b_var = 1 kg².",
          LEGEND, ""]
    for name, sc in SCENARIOS.items():
        agg, per_seed, meta = run_scenario(name)
        summary["scenarios"][name] = {"meta": meta, "aggregate": agg, "per_seed": per_seed}
        md.append(table(name, sc, agg))
        if not quiet:
            print(md[-1], flush=True)
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0
