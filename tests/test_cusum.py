"""The evidence-side channel: innovation record and per-sensor CUSUM.

(a) the recursion itself on a synthetic sequence; (b)-(d) what it sees and does
not see on the grid scenarios over 8 seeds, asserted with censored summaries.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments.phase1 import SCENARIOS, SPECS, constraint_for, declared_prior, run_scenario
from set_lcm.testbed.cusum import Cusum, CusumConfig, cusum_sequence
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import EstimatorSpec, run
from set_lcm.testbed.simulator import simulate

TEST_SEEDS = 8


@lru_cache(maxsize=None)
def agg(name: str) -> dict:
    return run_scenario(name, n_seeds=TEST_SEEDS)[0]


def _single(name: str, spec: EstimatorSpec):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    return run(PublicInputs.from_truth(truth), obs, constraint_for(truth), spec, m0, m0_std)


# ---------------------------------------------------------------------------
# (a) the recursion
# ---------------------------------------------------------------------------

def test_cusum_recursion_on_synthetic_sequence():
    """Zero-mean unit noise: no alarm in 1000 steps at h = 8 (two-sided ARL0 is ~1e4).
    A +1.5 shift from step 100: expected drift per step is 1.5 - k = 1.0, so the
    crossing of h = 8 comes within about ten steps."""
    rng = np.random.default_rng(3)
    cfg = CusumConfig(k=0.5, h=8.0)
    quiet = rng.normal(0.0, 1.0, 1000)
    stat, alarm = cusum_sequence(quiet, cfg)
    assert not alarm.any()
    assert stat.max() < cfg.h and stat.min() >= 0.0

    shifted = quiet.copy()
    shifted[100:] += 1.5
    stat, alarm = cusum_sequence(shifted, cfg)
    assert not alarm[:100].any()
    first = int(np.flatnonzero(alarm)[0])
    delay = first - 100
    assert 4 <= delay <= 15, delay
    assert stat[first] > cfg.h                     # the recorded value is the one that crossed
    assert stat[first + 1] < stat[first]           # and the channel was reset after the alarm
    # a persistent shift keeps re-alarming after every reset
    assert alarm[100:].sum() >= 10


def test_cusum_nan_leaves_channel_untouched_and_channels_are_independent():
    c = Cusum(CusumConfig(k=0.5, h=8.0), n=2)
    alarms = np.array([c.update(np.array([2.0, np.nan])) for _ in range(20)])   # only sensor 1 accumulates
    # +1.5 per update, alarm on the 6th (9 > 8) then reset: 20 = 6 + 6 + 6 + 2 -> stat 3.0, three alarms
    assert alarms[:, 0].sum() == 3 and c.stat[0] == pytest.approx(3.0)
    assert not alarms[:, 1].any()
    assert c.stat[1] == 0.0 and c.g_plus[1] == 0.0 and c.g_minus[1] == 0.0
    # the negative branch sees a downward shift
    c2 = Cusum(CusumConfig(k=0.5, h=8.0), n=1)
    alarms = [c2.update(np.array([-2.0]))[0] for _ in range(6)]
    assert alarms == [False, False, False, False, False, True]      # 6 x 1.5 = 9 > 8


# ---------------------------------------------------------------------------
# the record itself
# ---------------------------------------------------------------------------

def test_innovation_record_is_on_the_sampling_clock_and_alarms_on_the_report_clock():
    """bias_quant_delay has a 5-step arrival delay: the innovation of sample j is
    recorded at row j, the last five samples never arrive (NaN), and the alarm for a
    sample ingested at report step k is stamped at k."""
    spec = next(s for s in SPECS if s.name == "kf")
    rr = _single("bias_quant_delay", spec)
    n = len(rr.x)
    assert rr.innov.shape == rr.innov_var.shape == rr.innov_z.shape == (n, 2)
    assert rr.cusum_stat.shape == rr.cusum_alarm.shape == (n, 2) and rr.cusum_alarm.dtype == bool
    assert np.isnan(rr.innov[n - 5:]).all()                        # sampled, never arrived
    arrived = slice(0, n - 5)
    finite = np.isfinite(rr.innov_z[arrived])
    assert finite.mean() > 0.9                                     # 5 % dropout only
    np.testing.assert_allclose(rr.innov_z[arrived][finite], (rr.innov / np.sqrt(rr.innov_var))[arrived][finite])
    assert np.isnan(rr.cusum_stat[:5]).all() or (rr.cusum_stat[:5] == 0).all()   # nothing ingested yet
    # every alarm step has a statistic above h at that step (one ingest per step here)
    k = np.flatnonzero(rr.cusum_alarm.any(axis=1))
    assert k.size > 0
    assert (rr.cusum_stat[rr.cusum_alarm] > spec.cusum.h).all()
    # the constraint-side record is untouched by the channel
    off = replace(spec, name="kf(no cusum)", cusum=None)
    rr_off = _single("bias_quant_delay", off)
    assert np.array_equal(rr.x, rr_off.x) and np.array_equal(rr.stat, rr_off.stat)
    assert np.isnan(rr_off.cusum_stat).all() and not rr_off.cusum_alarm.any()


def test_hold_last_has_no_innovation_and_no_cusum():
    spec = next(s for s in SPECS if s.name == "hold_last")
    rr = _single("bias_quant_delay", spec)
    assert np.isnan(rr.innov_z).all()
    assert not rr.cusum_alarm.any()
    assert agg("bias_quant_delay")["hold_last"]["cusum"] is None


# ---------------------------------------------------------------------------
# (b)-(d) on the grid scenarios
# ---------------------------------------------------------------------------

def test_cusum_sees_the_sensor_bias_on_the_biased_sensor_only():
    """3 kg bias on sensor 1 from step 200, 5-step arrival delay: sensor 1's CUSUM
    alarms in every seed with a censored median delay under 20 steps (arrival delay
    included); sensor 2, which is unbiased, raises no alarm before onset."""
    c = agg("bias_quant_delay")["kf"]["cusum"]
    s1, s2 = c["s1"]["detection"], c["s2"]["detection"]
    assert s1["detected_any"] == s1["n"] == TEST_SEEDS
    assert not s1["median_censored"] and s1["median_delay"] < 20
    assert c["s2"]["false_alarms"]["max"] == 0
    assert c["s1"]["false_alarms"]["max"] == 0


def test_cusum_is_silent_on_the_closed_nominal_system():
    c = agg("closed_noise")["kf"]["cusum"]
    assert c["s1"]["false_alarms"]["max"] == 0
    assert c["s2"]["false_alarms"]["max"] == 0


def test_cusum_on_sensor_2_sees_the_leak():
    """The leak drains reservoir 2; the filter lags it and sensor 2's innovation goes
    negative. The channel sees it within 100 steps in most seeds, from the evidence
    alone and without reading the constraint."""
    c = agg("leak_stale_constraint")["kf"]["cusum"]
    s2 = c["s2"]["detection"]
    assert s2["detected_within"] >= 6 and s2["n"] == TEST_SEEDS
    # as for the guard, the honest null metric is the per-step rate, not "never":
    # at h = 8 one pre-leak alarm in 8 x 300 steps is within the channel's measured null
    assert c["s1"]["false_alarms"]["rate"] < 1e-3 and c["s2"]["false_alarms"]["rate"] < 1e-3


def test_cusum_is_identical_across_kf_variants():
    """Projection is a post-stage; the innovation is a filter property, so every
    kf-based variant carries the same channel."""
    a = agg("leak_stale_constraint")
    for est in ("kf+soft(1/lam=4)", "kf+hard", "kf+hard+guard"):
        assert a[est]["cusum"] == a["kf"]["cusum"], est
