"""A month of real six-minute water levels: the split, the resolution, and the stated sigma.

Reads the committed DAF observations (data/daf/noaa_8454000_202401_mllw.observations.json,
7,440 six-minute readings over January 2024), so nothing here needs a network or a DAF
checkout. The report is regenerated once per module and compared with the committed results
value for value, as tests/test_real_noaa.py does for the one-day report.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import real_noaa as day
from set_lcm.experiments import real_noaa_month as month
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.testbed.estimators_water import MONTH_CONSTITUENTS, TIDE_CONSTITUENTS, rayleigh_period_hours

RESULTS = REPO_ROOT / "results"


@pytest.fixture(scope="module")
def bridged():
    return month.load_month()


@pytest.fixture(scope="module")
def report():
    return month.compute()


# ---------------------------------------------------------------------------
# the evidence and the windows
# ---------------------------------------------------------------------------

def test_the_committed_month_bridges_to_a_complete_six_minute_grid(bridged):
    assert len(bridged.observations) == 31 * month.STEPS_PER_DAY
    t = np.asarray(bridged.inputs.t)
    assert np.all(np.diff(t) == 360.0)
    assert bridged.provenance["epoch_iso"] == "2024-01-01T00:00:00Z"
    assert all(bool(o.mask[0]) for o in bridged.observations)          # nothing missing


def test_an_edited_committed_file_cannot_reach_the_bridge(tmp_path, monkeypatch):
    """The manifest's sha256 is checked before the readings are used, not after."""
    man = month.manifest()
    entry = next(f for f in man["files"] if f["output"] == month.MONTH_FILE)
    monkeypatch.setitem(entry, "output_sha256", "0" * 64)
    with pytest.raises(ValueError, match="is not the manifest's"):
        month.manifest_entry(man)


def test_the_fit_and_held_out_windows_share_no_step_day_or_evidence_id(bridged):
    wins = {w.key: w for w in month.windows(len(bridged.observations))}
    month.check_disjoint(bridged, wins["fit"], wins["held_out"])      # raises if they do
    fit_ids = {i for o in bridged.observations[wins["fit"].start:wins["fit"].stop] for i in o.evidence_ids}
    held_ids = {i for o in bridged.observations[wins["held_out"].start:wins["held_out"].stop]
                for i in o.evidence_ids}
    assert fit_ids and held_ids and not (fit_ids & held_ids)
    assert wins["fit"].stop == wins["held_out"].start == month.FIT_DAYS * month.STEPS_PER_DAY
    assert wins["month"].n_steps == wins["fit"].n_steps + wins["held_out"].n_steps


def test_overlapping_windows_are_refused(bridged):
    a, b = month.Window("a", "a", 0, 500, "r"), month.Window("b", "b", 400, 900, "r")
    with pytest.raises(ValueError, match="overlap"):
        month.check_disjoint(bridged, a, b)


def test_a_sliced_window_keeps_the_records_absolute_clock(bridged):
    """A harmonic filter started mid-record must see the same phase reference as one at the
    start, so the slice keeps absolute seconds rather than rebasing to zero."""
    wins = {w.key: w for w in month.windows(len(bridged.observations))}
    inputs, obs = month.slice_window(bridged, wins["held_out"])
    t = np.asarray(bridged.inputs.t)
    assert inputs.t[0] == t[wins["held_out"].start] > 0.0
    assert len(obs) == wins["held_out"].n_steps
    assert obs[0] is bridged.observations[wins["held_out"].start]


# ---------------------------------------------------------------------------
# what the record's length can separate
# ---------------------------------------------------------------------------

def test_a_month_separates_what_a_day_cannot_and_the_half_does_not_separate_N2(report):
    resolution = {key: report["windows"][key]["resolution"] for key in ("fit", "held_out", "month")}
    assert resolution["month"]["unresolved_among_modelled"] == []
    unresolved = [tuple(row["pair"]) for row in resolution["fit"]["unresolved_among_modelled"]]
    assert unresolved == [("M2", "N2")], unresolved
    # and the criterion is the reason, computed rather than asserted
    assert rayleigh_period_hours("M2", "N2") / 24.0 > resolution["fit"]["record_days"]
    assert rayleigh_period_hours("M2", "N2") / 24.0 <= resolution["month"]["record_days"]
    for pair in (("M2", "S2"), ("K1", "O1")):
        assert rayleigh_period_hours(*pair) / 24.0 <= resolution["fit"]["record_days"]
    # a day cannot separate any of the three, which is why the day report models four
    for pair in (("M2", "S2"), ("M2", "N2"), ("K1", "O1")):
        assert rayleigh_period_hours(*pair) > 24.0
    assert set(TIDE_CONSTITUENTS) < set(MONTH_CONSTITUENTS)


def test_P1_is_unresolved_at_a_month_and_modelled_nowhere(report):
    for key in ("fit", "held_out", "month"):
        resolution = report["windows"][key]["resolution"]
        pairs = [tuple(row["pair"]) for row in resolution["unresolved"]]
        assert ("K1", "P1") in pairs, key
        assert "P1" not in resolution["modelled_by_tide_month"]
        assert "P1" not in resolution["modelled_by_tide_kf"]
    assert rayleigh_period_hours("K1", "P1") / 24.0 > 180.0
    for kind, declared in report["declared"]["constituents"].items():
        assert "P1" not in declared, kind


# ---------------------------------------------------------------------------
# the fit, and the two scorings of the held-out half
# ---------------------------------------------------------------------------

def test_q_is_fitted_on_the_fit_window_and_nowhere_else(report, bridged):
    wins = {w.key: w for w in month.windows(len(bridged.observations))}
    for kind, fit in report["fits"].items():
        assert fit["window"] == "fit"
        assert fit["n_steps"] == wins["fit"].n_steps
        assert [p["q_scale"] for p in fit["profile"]] == list(month.GRIDS[kind])
        best = max(fit["profile"], key=lambda p: p["loglik"])
        assert fit["q_scale"] == best["q_scale"] and fit["loglik"] == best["loglik"]
        assert fit["interior"] == (0 < fit["argmax_index"] < len(fit["profile"]) - 1)


def test_the_declared_grids_are_the_day_reports(report):
    assert report["declared"]["q_grids"]["level_trend"] == list(day.Q_GRIDS["level_trend"])
    assert report["declared"]["q_grids"]["tide_kf"] == list(day.Q_GRIDS["tide_kf"])
    assert report["declared"]["q_grids"]["tide_month"] == list(day.Q_GRIDS["tide_kf"])


def test_both_scorings_read_the_same_held_out_readings(report, bridged):
    """They differ in the state entering the window, not in the evidence."""
    wins = {w.key: w for w in month.windows(len(bridged.observations))}
    for kind, scored in report["scored"].items():
        fresh, continued = scored["held_out"]["fresh"], scored["held_out"]["continued"]
        assert fresh["n_steps"] == continued["n_steps"] == wins["held_out"].n_steps
        assert fresh["n_observed"] == continued["n_observed"]
        assert fresh["n_evidence_ids"] == continued["n_evidence_ids"]
        assert fresh["q_scale"] == continued["q_scale"] == report["fits"][kind]["q_scale"]


def test_the_continued_alarm_step_is_reported_in_both_coordinate_systems(report, bridged):
    """truth_free reports an alarm step in its run's own clock, and the two runs differ.

    Comparing the fresh and continued scorings would otherwise mean silently subtracting an
    offset, so the continued entry carries both forms and they must agree.
    """
    offset = month.FIT_DAYS * month.STEPS_PER_DAY
    n_steps = len(bridged.observations)
    for kind, scored in report["scored"].items():
        continued = scored["held_out"]["continued"]
        in_record, in_window = continued["cusum_first_alarm_in_record"], continued["cusum_first_alarm"]
        if in_record is None:
            assert in_window is None
            continue
        assert in_window == in_record - offset, kind
        assert 0 <= in_window < n_steps - offset
        assert in_record >= offset          # the alarm is inside the held-out window


def test_the_burn_in_null_is_reported_as_a_measurement(report):
    """A cold start matches a continued run here. The report must say so, not omit it."""
    for kind, scored in report["scored"].items():
        ratio = scored["held_out"]["z_rms_fresh_over_continued"]
        assert ratio is not None and 0.9 < ratio < 1.1, (kind, ratio)
    text = month.render(report)
    assert "The burn-in buys nothing measurable here" in text
    assert "not that carried state does not matter" in text


# ---------------------------------------------------------------------------
# the stated sigma, which one reading dominates
# ---------------------------------------------------------------------------

def test_the_stated_sigma_is_described_by_its_median_not_only_its_rms(report, bridged):
    sigma = np.array([float(o.R[0, 0]) ** 0.5 for o in bridged.observations])
    described = report["windows"]["month"]["stated_sigma"]
    assert described["median"] == pytest.approx(float(np.median(sigma)), rel=0, abs=0)
    assert described["rms"] == pytest.approx(float(np.sqrt(np.mean(sigma ** 2))), rel=1e-12)
    assert described["max"] == pytest.approx(float(sigma.max()), rel=0, abs=0)
    # the single largest statement, and what removing it does to a second moment
    assert described["n_extreme"] >= 1
    worst = described["extreme"][0]
    assert worst["sigma"] == described["max"] and worst["over_median"] > 100.0
    assert bridged.observations[worst["step_in_record"]].R[0, 0] == pytest.approx(worst["sigma"] ** 2)
    assert described["rms_without_extreme"] < described["rms"] / 2.0
    assert described["n_stated_zero"] == int(np.count_nonzero(sigma == 0.0)) > 0


def test_the_report_names_the_one_reading_that_drives_every_second_moment(report):
    text = month.render(report)
    described = report["windows"]["month"]["stated_sigma"]
    assert f"{described['max']:.3f} m" in text
    assert f"{described['median']:.4f}" in text
    assert "second moment" in text and "median is unchanged by it" in text
    assert f"{described['n_stated_zero']} readings state sigma = 0.000 m" in text


def test_the_stated_sigma_exceeds_the_white_error_bound_in_every_window(report):
    """A real finding, stated with the assumption that lets it be read correctly."""
    for key in ("fit", "held_out", "month"):
        window = report["windows"][key]
        ratio = window["stated_sigma_over_white_bound"]
        check = window["series_check"]
        assert ratio["median"] == pytest.approx(
            check["median_stated_sigma"] / check["white_error_sigma_bound"], rel=1e-12)
        assert ratio["median"] > 1.0, key
    text = month.render(report)
    assert "exceeds the white-error bound in every window" in text
    assert "nothing here shows the declared R to be wrong" in json.dumps(report["limitations"])
    assert "smoother than the declared R" in text


def test_the_model_free_bound_is_sharper_over_a_month_than_over_a_half(report):
    """31 days of second differences scatter less than 15, which is what the month buys."""
    counts = {key: report["windows"][key]["series_check"]["n_second_differences"]
              for key in ("fit", "held_out", "month")}
    assert counts["month"] > counts["fit"] and counts["month"] > counts["held_out"]
    # Each window of n steps has n - 2 second differences, so the month has the two that
    # straddle the split and the halves do not.
    assert counts["month"] == counts["fit"] + counts["held_out"] + 2


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------

def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "real_noaa_month.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("real_noaa_month.json", report, committed)
    assert failure is None, failure
    assert month.render(committed) == (RESULTS / "real_noaa_month.md").read_text(encoding="utf-8")


def test_the_report_says_what_it_cannot_validate(report):
    text = month.render(report)
    assert "## What these numbers do not show" in text
    for phrase in ("Truth-free", "no number here is an error", "not a calibration",
                   "The split is not free", "absorbed into the K1 coefficients"):
        assert phrase in text, phrase
    assert report["windows"]["month"]["role"].endswith("in-sample for every fitted q")


# ---------------------------------------------------------------------------
# the report is not written when its own sentences stop being true
# ---------------------------------------------------------------------------

def test_every_qualitative_sentence_is_a_computed_condition(report):
    assert report["claims"], "the report declares no claims"
    assert all(report["claims"].values()), [k for k, v in report["claims"].items() if not v]


def test_a_failed_claim_refuses_to_write_the_report(report):
    """The guard exists because a sentence behind `if computable` is not a sentence behind
    `if true`: the first draft printed "a z RMS below 1" when 11 of 12 scorings were."""
    broken = dict(report)
    broken["claims"] = dict(report["claims"], no_missing_readings=False)
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        month.render(broken)


def test_the_z_rms_sentence_counts_the_scorings_rather_than_asserting_all_of_them(report):
    scorings = [entry[key]["z_rms"] for entry in report["scored"].values()
                for key in ("in_sample_fit_window", "whole_month_in_sample")]
    scorings += [entry["held_out"][key]["z_rms"] for entry in report["scored"].values()
                 for key in ("fresh", "continued")]
    below = sum(1 for value in scorings if value < 1.0)
    assert 0 < below < len(scorings), "the scoping only matters while not every scoring is below 1"
    assert f"which {below} of the {len(scorings)} scorings above are" in month.render(report)


def test_the_claims_cover_the_sentences_that_assert_something_about_the_data(report):
    """A spot check that the guard is not decorative: each of these is a sentence in the text."""
    expected = {
        "month_separates_every_modelled_pair",
        "fit_window_leaves_a_modelled_pair_unresolved",
        "fit_window_separates_S2_and_O1",
        "P1_unresolved_in_every_window",
        "P1_modelled_nowhere",
        "fit_and_held_out_share_no_evidence",
        "burn_in_changes_z_rms_by_under_five_percent",
        "continued_alarm_steps_convert_to_the_fresh_window",
        "one_reading_dominates_the_stated_sigma_rms",
        "the_largest_stated_sigma_is_in_the_held_out_half",
        "stated_sigma_exceeds_the_white_bound_in_every_window",
    }
    assert expected <= set(report["claims"]), expected - set(report["claims"])
