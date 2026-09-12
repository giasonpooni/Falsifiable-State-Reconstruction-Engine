"""The truth-free evaluator: evaluate_truth_free(run, windows) reads only the RunResult.

On simulated records, where the truth is available to check it against, over 8
seeds: what it reports on the nominal system (and what that says about a known
mis-statement of Q), what it reports when a sensor acquires an undeclared bias,
and that its mean z is exactly the z̄ column the truth evaluator already reports;
over 2 seeds, that its mean consistency statistic is the calibration module's
in-loop null.
"""
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments import calibration
from set_lcm.experiments.phase1 import SCENARIOS, SPECS, constraint_for, declared_prior, iter_runs
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.evaluate import evaluate
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import run
from set_lcm.testbed.simulator import simulate
from set_lcm.testbed.truth_free import evaluate_truth_free

TEST_SEEDS = 8
KF = next(s for s in SPECS if s.name == "kf")
BIAS_WINDOWS = {"pre_bias": (0, 200), "onset": (200, 260), "post_bias": (200, 600)}


@lru_cache(maxsize=None)
def _runs(name: str, windows: tuple) -> list[tuple[dict, dict]]:
    """(truth-free report, truth evaluator report) per seed, for kf."""
    sc = SCENARIOS[name]
    out = []
    for _, truth, _, cs, rr in iter_runs(name, KF, TEST_SEEDS):
        out.append((evaluate_truth_free(rr, dict(windows)), evaluate(rr, truth, sc.fault_onset, sc.windows, cs=cs)))
    return out


def _sensor(reports, window: str, i: int, key: str) -> list:
    return [tf["windows"][window]["sensors"][i][key] for tf, _ in reports]


def test_closed_noise_innovations_are_centred_and_show_the_q_overstatement_only_faintly():
    """closed_noise, kf, 8 seeds, steady window (300-600), from the record alone:
    z̄ = +0.013 / -0.008 and RMS z = 0.97 / 0.98 for sensors 1 / 2 (lag-1 +0.01 / -0.00,
    |z| > 1.96 in 4.5 % / 4.3 % of samples, no CUSUM alarm, no flag).

    RMS z a few percent below 1 is the truth-free face of the known Q over-statement.
    KFConfig's diagonal Q asserts 0.05 kg of process noise per step and reservoir that
    the closed simulator does not generate with the pump off: in steady state the filter
    states a prediction variance of about 0.10 kg^2 per reservoir where its actual
    prediction error variance is about 0.05 kg^2. Against truth that factor of two reads
    nz = 0.73 (results/summary.md); in the consistency statistic, which it enters
    undiluted, it is a mean of 0.66 here (0.46-0.64 over the nominal windows of
    results/calibration.md). In the innovation it is diluted by R = 4 kg^2: S is stated
    as 4.10 kg^2 against an actual 4.05, so RMS z = sqrt(4.05 / 4.10) = 0.994 analytically,
    which 8 seeds x ~285 samples per sensor (standard error ~0.015) barely resolve. On a
    real record, per-sensor innovations that look calibrated say little about Q when R
    dominates S; the constraint statistic is the sharper witness."""
    reports = _runs("closed_noise", (("steady", (300, 600)),))
    for i in range(2):
        assert abs(np.mean(_sensor(reports, "steady", i, "z_mean"))) < 0.02
        assert 0.95 < np.mean(_sensor(reports, "steady", i, "z_rms")) < 1.0
        assert abs(np.mean(_sensor(reports, "steady", i, "z_lag1"))) < 0.05
        assert 0.03 < np.mean(_sensor(reports, "steady", i, "frac_abs_z_gt_1.96")) < 0.07
        assert 0.03 < np.mean(_sensor(reports, "steady", i, "frac_missing")) < 0.08      # 5 % dropout
        assert _sensor(reports, "steady", i, "cusum_alarms") == [0] * TEST_SEEDS
    con = [tf["windows"]["steady"]["constraint"] for tf, _ in reports]
    assert all(c["flag_steps"] == 0 and c["status_counts"] == {"skipped": 300} for c in con)
    assert np.mean([c["stat_mean"] for c in con]) < 0.8          # the undiluted face: well below 1
    assert all(tf["windows"]["steady"]["n_evidence_ids"] == 0 for tf, _ in reports)
    # its z̄ is exactly the truth evaluator's z̄ column, which never needed the truth either
    for tf, ev in reports:
        assert [s["z_mean"] for s in tf["windows"]["steady"]["sensors"]] == ev["innov_z_mean_by_window"]["steady"]


def test_bias_shows_as_a_shift_of_sensor_1_mean_z():
    """bias_quant_delay, kf, 8 seeds: sensor 1 gains an undeclared +3 kg at step 200
    (0.5 kg quantization, 5-step arrival delay). From the record alone sensor 1's mean z
    moves from +0.01 before onset to +0.77 over the first 60 samples after it (every seed
    above +0.6) and +0.15 over the whole post-bias window -- the filter absorbs the offset
    into m1, so the shift decays -- while sensor 2 stays within 0.02 of 0. Sensor 1's
    CUSUM alarms in the first 60 samples in every seed, sensor 2's never. The per-seed z̄
    over the scenario's windows is exactly evaluate()'s (the z̄ s1/s2 column of
    results/summary.md: -0.01 / +0.16 pre / post over 20 seeds). What the record cannot
    say: whether sensor 1 is biased or reservoir 1 really gained 3 kg the model does not
    explain -- both write the same record."""
    reports = _runs("bias_quant_delay", tuple(BIAS_WINDOWS.items()))
    s1 = {w: _sensor(reports, w, 0, "z_mean") for w in BIAS_WINDOWS}
    s2 = {w: _sensor(reports, w, 1, "z_mean") for w in BIAS_WINDOWS}
    assert abs(np.mean(s1["pre_bias"])) < 0.05
    assert min(s1["onset"]) > 0.5
    assert min(s1["post_bias"]) > 0.1 and np.mean(s1["post_bias"]) > np.mean(s1["pre_bias"]) + 0.1
    for w in BIAS_WINDOWS:
        assert abs(np.mean(s2[w])) < 0.05, w
    assert _sensor(reports, "pre_bias", 0, "cusum_alarms") == [0] * TEST_SEEDS
    assert all(a >= 1 for a in _sensor(reports, "onset", 0, "cusum_alarms"))
    assert all(200 < k < 270 for k in _sensor(reports, "onset", 0, "cusum_first_alarm"))  # report clock: + delay
    for w in BIAS_WINDOWS:
        assert _sensor(reports, w, 1, "cusum_alarms") == [0] * TEST_SEEDS, w
    # the consistency flag, read from the same record: silent before, raised after
    assert all(tf["windows"]["pre_bias"]["constraint"]["flag_steps"] == 0 for tf, _ in reports)
    assert all(tf["windows"]["post_bias"]["constraint"]["flag_steps"] > 0 for tf, _ in reports)
    # the last 5 samples never arrive (5-step delay): missing from what the estimator was given
    assert all(tf["windows"]["post_bias"]["sensors"][1]["frac_missing"] >= 5 / 400 for tf, _ in reports)
    for tf, ev in reports:
        for w in ("pre_bias", "post_bias"):
            assert [s["z_mean"] for s in tf["windows"][w]["sensors"]] == ev["innov_z_mean_by_window"][w], w


def test_consistency_statistic_mean_is_the_calibration_null_from_the_record_alone():
    """The in-loop null of results/calibration.md (mean 0.46-0.64 over the four nominal
    windows, 20 seeds) is a truth-free quantity: the truth-free evaluator's per-seed mean
    consistency statistic, averaged over seeds, reproduces calibration.null_stats. Checked
    here at 2 seeds; equal sample counts per seed make the pooled mean the mean of means."""
    ref = calibration.null_stats(n_seeds=2)
    for name, (lo, hi) in calibration.NULL_WINDOWS.items():
        tf = [evaluate_truth_free(rr, {"w": (lo, hi)})["windows"]["w"]["constraint"]["stat_mean"]
              for *_, rr in iter_runs(name, calibration.KF, 2)]
        assert np.mean(tf) == pytest.approx(ref[name]["mean"], rel=1e-12), name


def test_what_is_not_applicable_is_reported_as_such():
    """hold-last has observations but no innovation (z and CUSUM None); a run with no
    declared constraint has no constraint block; kf_aug's parameter flags are counted;
    windows outside the run are refused."""
    sc = SCENARIOS["leak_stale_constraint"]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    pi = PublicInputs.from_truth(truth)
    wins = {"pre_leak": (0, 300), "leak_and_after": (300, 600)}

    hl = evaluate_truth_free(run(pi, obs, constraint_for(truth), next(s for s in SPECS if s.name == "hold_last"),
                                 m0, m0_std), wins)
    s = hl["windows"]["pre_leak"]["sensors"][0]
    assert s["n_observed"] > 250 and s["n_z"] == 0
    assert s["z_mean"] is s["z_rms"] is s["z_lag1"] is s["cusum_alarms"] is None
    assert not hl["cusum_applicable"]

    free = evaluate_truth_free(run(pi, obs, None, KF, m0, m0_std), wins)
    assert not free["constraint_declared"] and free["windows"]["pre_leak"]["constraint"] is None
    assert free["windows"]["pre_leak"]["extra_flags"] is None
    assert free["n_sensors"] == 2 and free["n_report"] == 2

    aug = evaluate_truth_free(run(pi, obs, constraint_for(truth), next(s for s in SPECS if s.name == "kf_aug"),
                                  m0, m0_std), wins)
    flags = aug["windows"]["leak_and_after"]["extra_flags"]
    assert set(flags) == {"alpha", "L"}
    assert aug["windows"]["pre_leak"]["extra_flags"]["L"]["flag_steps"] == 0
    assert flags["L"]["flag_steps"] > 0 and flags["L"]["first_flag_step"] >= 300

    for bad in ({"w": (0, 601)}, {"w": (300, 300)}, {"w": (-1, 10)}):
        with pytest.raises(ValueError):
            evaluate_truth_free(run(pi, obs, None, KF, m0, m0_std), bad)
