"""P4: real NOAA water levels through the same runner (results/real_noaa.{md,json}).

The run is small -- two water-level filters, a 13-point q grid, three days of 240 readings,
a few seconds -- so the fast suite regenerates it and compares it with the committed files
value for value: the JSON with only the latency columns and the generation stamp (interpreter,
platform, source hash, git HEAD) excluded, as tests/test_results_reproduce.py does for the
grid, and the markdown with only its generation line excluded. It also pins that the q_scale
search never saw the held-out day, that the datum-invariance numbers in the report are what an
independent pass through run() computes, and what the water-level filters refuse.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import real_noaa
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.schema import Observation
from set_lcm.testbed.estimators import ESTIMATORS, KFConfig
from set_lcm.testbed.estimators_water import LevelTrendConfig, TideConfig, TideKF, speed_rad_per_s
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import EstimatorSpec, run

RESULTS = REPO_ROOT / "results"
COMMITTED = json.loads((RESULTS / "real_noaa.json").read_text(encoding="utf-8"))
COMMITTED_MD = (RESULTS / "real_noaa.md").read_text(encoding="utf-8")


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "generation" and not k.startswith("latency_us")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _body(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if not ln.startswith("Generated with ")]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("real_noaa")
    real_noaa.main(out, quiet=True)
    return (json.loads((out / "real_noaa.json").read_text(encoding="utf-8")),
            (out / "real_noaa.md").read_text(encoding="utf-8"))


def test_real_noaa_report_reproduces(fresh):
    data, md = fresh
    assert set(data["provenance"]["generation"]) == set(COMMITTED["provenance"]["generation"])
    # the JSON: bitwise on the build that generated it, the declared cross-build tolerance
    # elsewhere (set_lcm.experiments.compare says what was measured and why)
    failure = reproduction_failure("real_noaa.json", data, COMMITTED)
    assert failure is None, failure
    # the report itself: every number it states, at the precision it states it, on every build
    assert _body(md) == _body(COMMITTED_MD), "results/real_noaa.md does not match a fresh run of the source tree"
    # what the provenance block must carry: DAF's commit, each day's bridge provenance, the source hash
    prov = COMMITTED["provenance"]
    assert prov["daf_commit"] == json.loads((REPO_ROOT / "data" / "daf" / "manifest.json").read_text())["daf_commit"]
    for key in ("fit", "held_out", "stnd"):
        b = prov["bridge"][key]
        assert b["daf_commit"] == prov["daf_commit"] and len(b["evidence_ids_sha256"]) == 64
        assert (b["n_records_in"], b["n_used"], b["n_in_conflict"]) == (240, 240, 0)
    assert len(prov["generation"]["source_sha256"]) == 64
    assert all(COMMITTED["claims"].values())


def test_the_q_search_never_sees_the_held_out_day(monkeypatch):
    """q_scale is fitted on 2024-01-15 MLLW only; the held-out day is a different day with no
    evidence id in common, and every run on it uses exactly the fitted value."""
    fits, runs = [], []
    fit_q, run_one = real_noaa.fit_q, real_noaa.run_one

    def fit_spy(kind, bs):
        fits.append((kind, bs.provenance["epoch_iso"], bs.source_ids))
        return fit_q(kind, bs)

    def run_spy(kind, q, bs, r_scale=1.0):
        runs.append((kind, q, bs.provenance["epoch_iso"], bs.source_ids[0], r_scale))
        return run_one(kind, q, bs, r_scale)

    monkeypatch.setattr(real_noaa, "fit_q", fit_spy)
    monkeypatch.setattr(real_noaa, "run_one", run_spy)
    r = real_noaa.compute()
    assert fits == [(k, "2024-01-15T00:00:00Z", ("noaa:8454000:MLLW:m",)) for k in real_noaa.KINDS]
    for k in real_noaa.KINDS:
        q_hat = r["fit"][k]["q_scale"]
        assert r["fit"][k]["day"] == "fit"
        assert {q for kind, q, epoch, _, _ in runs if kind == k and epoch.startswith("2026-08-23")} == {q_hat}
        fit_day_mllw = [q for kind, q, epoch, src, s in runs
                        if kind == k and epoch.startswith("2024-01-15") and src.endswith(":MLLW:m") and s == 1.0]
        assert set(real_noaa.Q_GRIDS[k]) <= set(fit_day_mllw)
        assert {q for kind, q, _, src, _ in runs if kind == k and src.endswith(":STND:m")} == {q_hat}
        assert r["fit"][k]["evidence_ids_sha256"] == r["days"]["fit"]["evidence_ids_sha256"]
    fit_day, held = r["days"]["fit"], r["days"]["held_out"]
    assert fit_day["first_grid_point"][:10] == "2024-01-15" and held["first_grid_point"][:10] == "2026-08-23"
    assert fit_day["last_grid_point"] < held["first_grid_point"]
    bs_fit, bs_held = real_noaa.load_day(real_noaa.FIT), real_noaa.load_day(real_noaa.HELD_OUT)
    assert not set(bs_fit.evidence_ids_used) & set(bs_held.evidence_ids_used)
    real_noaa.check_disjoint(bs_fit, bs_held)
    with pytest.raises(ValueError):
        real_noaa.check_disjoint(bs_fit, bs_fit)
    with pytest.raises(ValueError):                  # the same day on another datum is not another window
        real_noaa.check_disjoint(bs_fit, real_noaa.load_day(real_noaa.DATUM))


def test_datum_invariance_numbers_match_the_report():
    """An independent pass through run(): 2024-01-15 on MLLW and on STND with the committed
    q_scale gives the report's numbers exactly, and the report prints them."""
    bs_m, bs_s = real_noaa.load_day(real_noaa.FIT), real_noaa.load_day(real_noaa.DATUM)
    y_m = np.array([o.y[0] for o in bs_m.observations])
    y_s = np.array([o.y[0] for o in bs_s.observations])
    np.testing.assert_allclose(y_s - y_m, 1.064, atol=1e-9)          # the same surface, a constant apart
    cfgs = {"level_trend": LevelTrendConfig, "tide_kf": TideConfig}
    for kind in ("level_trend", "tide_kf"):
        q = COMMITTED["fit"][kind]["q_scale"]
        spec = EstimatorSpec(kind, kind, None)
        rm = run(bs_m.inputs, bs_m.observations, None, spec, (0.0,), 10.0, est_cfg=cfgs[kind](q_scale=q))
        rs = run(bs_s.inputs, bs_s.observations, None, spec, (0.0,), 10.0, est_cfg=cfgs[kind](q_scale=q))
        dz = np.abs(rm.innov_z[:, 0] - rs.innov_z[:, 0])
        state = (lambda rr: rr.x[:, 0]) if kind == "level_trend" else (lambda rr: rr.extra["mean_level_hat"])
        off = state(rs) - state(rm)
        rep = COMMITTED["datum_invariance"][kind]
        assert rep["burn_in_steps"] == 20
        # These small diagnostics are cancellation-sensitive across numerical builds.
        # Observed absolute difference: 1.2e-13. Keep the winning step exact and use
        # tight absolute/relative bounds for the arithmetic, including near-zero values.
        close = lambda value: pytest.approx(value, rel=1e-8, abs=1e-11)
        assert rep["max_abs_dz_after_burn_in"] == close(float(dz[20:].max()))
        assert rep["argmax_step_after_burn_in"] == 20 + int(np.argmax(dz[20:]))
        assert rep["max_abs_dz_all_steps"] == close(float(dz.max()))
        assert rep["max_abs_dz_from_step_1"] == close(float(dz[1:].max()))
        assert rep["offset_final"] == close(float(off[-1]))
        assert (rep["offset_min_after_burn_in"], rep["offset_max_after_burn_in"]) == \
            close((float(off[20:].min()), float(off[20:].max())))
        row = (f"| {kind} | {rep['state']} | {rep['max_abs_dz_after_burn_in']:.2e} "
               f"({rep['argmax_step_after_burn_in']}) | {rep['max_abs_dz_all_steps']:.4f} | {rep['offset_final']:.6f} |")
        assert row in COMMITTED_MD
    # what the report says about them
    assert COMMITTED["datum_invariance"]["level_trend"]["max_abs_dz_from_step_1"] < 1e-4
    assert COMMITTED["datum_invariance"]["level_trend"]["max_abs_dz_after_burn_in"] < 1e-6
    assert abs(COMMITTED["datum_invariance"]["tide_kf"]["offset_final"] - 1.064) < 1e-4
    assert COMMITTED["datum_invariance"]["tide_kf"]["offset_min_after_burn_in"] < 1.054


# ---------------------------------------------------------------------------
# the water-level filters
# ---------------------------------------------------------------------------

def _record(n=60, dt=360.0, sensors=1, t=None, u=None):
    t = np.arange(n) * dt if t is None else t
    obs = [Observation(float(t[j]), float(t[j]), np.full(sensors, 0.5), np.eye(sensors) * 1e-4,
                       np.ones(sensors, dtype=bool), tuple(f"s{i}" for i in range(sensors))) for j in range(n)]
    return PublicInputs(t=t, u_commanded=np.zeros(n) if u is None else u), obs


def test_water_filters_refuse_what_they_cannot_model():
    pi, obs = _record()
    lt = EstimatorSpec("level_trend", "level_trend", None)
    with pytest.raises(ValueError):
        run(pi, obs, None, lt, (0.0,), 10.0)                                    # q_scale has no default
    with pytest.raises(ValueError):
        run(pi, obs, None, lt, (0.0,), 10.0, est_cfg=TideConfig(q_scale=1e-4))  # the wrong config
    with pytest.raises(ValueError):                                            # kf keeps its own keyword
        run(pi, obs, None, EstimatorSpec("kf", "kf", None), (0.0, 0.0), 1.0, est_cfg=KFConfig())
    for bad in (0.0, -1e-6, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            LevelTrendConfig(q_scale=bad)
        with pytest.raises(ValueError):
            TideConfig(q_scale=bad)
    for kind, cfg in (("level_trend", LevelTrendConfig(q_scale=1e-6)), ("tide_kf", TideConfig(q_scale=1e-4))):
        spec = EstimatorSpec(kind, kind, None)
        pi2, obs2 = _record(sensors=2)                                           # two datums are two quantities
        with pytest.raises(ValueError):
            run(pi2, obs2, None, spec, (0.0,), 10.0, est_cfg=cfg)
        pi3, obs3 = _record(u=np.r_[np.zeros(30), np.ones(30)])                 # a tide gauge commands nothing
        with pytest.raises(ValueError):
            run(pi3, obs3, None, spec, (0.0,), 10.0, est_cfg=cfg)
        t = np.arange(60) * 360.0
        t[30:] += 60.0                                                           # one dt from the grid, or none
        pi4, obs4 = _record(t=t)
        with pytest.raises(ValueError):
            run(pi4, obs4, None, spec, (0.0,), 10.0, est_cfg=cfg)
        with pytest.raises(ValueError):
            run(pi, obs, None, spec, (0.0, 0.0), 10.0, est_cfg=cfg)             # the prior is one level
        rr = run(pi, obs, None, spec, (0.0,), 10.0, est_cfg=cfg)
        assert rr.x.shape == (60, 1) and rr.threshold is None
        assert {s.value for s in rr.status} == {"skipped"}                      # no constraint, nothing projected
        assert ESTIMATORS[kind].aug_nominal == (None,) * len(ESTIMATORS[kind].aug_names)
        assert not any(key.startswith("flag_") for key in rr.extra)             # recorded, never flagged
        with pytest.raises(NotImplementedError):
            ESTIMATORS[kind]((0.0,), 10.0, 360.0, pi.u_commanded, cfg, clock=pi.t).set_state(np.zeros(1), np.eye(1))


def test_tide_kf_reports_the_level_its_states_imply():
    """x[:, 0] is H_k x_k: the mean level plus each constituent's a cos(w t_k) + b sin(w t_k)."""
    bs = real_noaa.load_day(real_noaa.FIT)
    rr = run(bs.inputs, bs.observations, None, EstimatorSpec("tide_kf", "tide_kf", None), (0.0,), 10.0,
             est_cfg=TideConfig(q_scale=COMMITTED["fit"]["tide_kf"]["q_scale"]))
    assert TideKF.aug_names == ("mean_level", "M2_a", "M2_b", "K1_a", "K1_b", "O1_a", "O1_b", "M4_a", "M4_b")
    level = rr.extra["mean_level_hat"].copy()
    for c in ("M2", "K1", "O1", "M4"):
        w = speed_rad_per_s(c)
        level += rr.extra[f"{c}_a_hat"] * np.cos(w * bs.inputs.t) + rr.extra[f"{c}_b_hat"] * np.sin(w * bs.inputs.t)
    np.testing.assert_allclose(rr.x[:, 0], level, rtol=0, atol=1e-12)
    assert speed_rad_per_s("M2") == pytest.approx(2 * np.pi / (12.4206012 * 3600), rel=1e-7)   # 12.42 h period
    # the reported level variance h P h^T: positive, except where NOAA stated sigma = 0.000 (R = 0,
    # passed through by the bridge): there the level is stated exactly and its variance is 0 up to
    # the roundoff of the 10 x 10 transform, which may fall either side of 0
    var = rr.P[:, 0, 0]
    zero_r = np.array([o.R[0, 0] == 0.0 for o in bs.observations])
    assert zero_r.sum() == 2 and np.all(var[~zero_r] > 0.0) and np.all(np.abs(var[zero_r]) < 1e-12)
    assert rr.ingested_evidence == [o.evidence_ids for o in bs.observations]


def test_level_trend_follows_a_ramp():
    """A synthetic ramp (0.3 m + 2e-5 m/s, white noise of 1 mm, R = 1e-6 m^2): the rate state
    converges to the slope within its own reported uncertainty."""
    rng = np.random.default_rng(3)
    n = 200
    t = np.arange(n) * 360.0
    y = 0.3 + 2e-5 * t + rng.normal(0.0, 1e-3, n)
    obs = [Observation(float(t[j]), float(t[j]), np.array([y[j]]), np.array([[1e-6]]), np.array([True]), ("s",))
           for j in range(n)]
    rr = run(PublicInputs(t=t, u_commanded=np.zeros(n)), obs, None, EstimatorSpec("lt", "level_trend", None),
             (0.0,), 10.0, est_cfg=LevelTrendConfig(q_scale=1e-8))
    assert abs(rr.extra["rate_hat"][-1] - 2e-5) < 3 * rr.extra["rate_sd"][-1]
    assert abs(rr.x[-1, 0] - (0.3 + 2e-5 * t[-1])) < 3 * np.sqrt(rr.P[-1, 0, 0])
    assert set(rr.extra) == {"rate_hat", "rate_sd"}                              # nominal None: no flag
