"""Strict timestamp precision, identity-preserving alignment, and boundary checks."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from set_lcm.recording_clock import compare_capture_clocks, parse_capture_utc


@pytest.mark.parametrize("text,expected", [
    ("2026-09-12T12:34:56Z", datetime(2026, 9, 12, 12, 34, 56, tzinfo=timezone.utc)),
    ("2026-09-12T08:34:56-04:00", datetime(2026, 9, 12, 12, 34, 56, tzinfo=timezone.utc)),
    ("2026-09-12T18:04:56+05:30", datetime(2026, 9, 12, 12, 34, 56, tzinfo=timezone.utc)),
    ("2026-09-12T00:00:00+00:30", datetime(2026, 9, 11, 23, 30, tzinfo=timezone.utc)),
    ("2026-09-12T12:34:56.123456Z", datetime(2026, 9, 12, 12, 34, 56, 123456, tzinfo=timezone.utc)),
    ("2026-09-12T12:34:56,1+00:00", datetime(2026, 9, 12, 12, 34, 56, 100000, tzinfo=timezone.utc)),
    ("2024-02-29T23:59:59-00:00", datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc)),
])
def test_explicit_instants_normalize_offsets_without_losing_accepted_precision(text, expected):
    actual = parse_capture_utc(text)
    assert actual == expected
    assert actual.tzinfo is timezone.utc


@pytest.mark.parametrize("text", [
    None, "", " ", 0, np.nan, np.inf, True,
    "2026-09-12", "2026-09-12T12:34:56", "2026-09-12T12:34Z",
    "20260912T123456Z", "2026-09-12 12:34:56Z", "2026-09-12T12:34:56 UTC",
    "2026-09-12T12:34:56Z ", " 2026-09-12T12:34:56Z",
    "2026-02-29T12:34:56Z", "2026-09-12T24:00:00Z", "2026-09-12T12:60:00Z",
    "2026-09-12T12:34:60Z", "2026-09-12T12:34:56+24:00", "2026-09-12T12:34:56+00:60",
    "2026-09-12T12:34:56+01:00:00", "2026-09-12T12:34:56.Z",
    "2026-09-12T12:34:56.1234567Z", "2026-09-12T12:34:56.1234560+05:30",
    "2026-09-12T12:34:56,0000001-04:00", "2026-09-12T12:34:56.123456789123456789Z",
    "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00",
])
def test_absent_naive_invalid_or_unsupported_timestamps_are_refused(text):
    with pytest.raises(ValueError):
        parse_capture_utc(text)


def test_more_than_six_fraction_digits_are_refused_even_if_extra_digits_are_zero():
    with pytest.raises(ValueError, match="not truncated"):
        parse_capture_utc("2026-09-12T12:34:56.1000000Z")


def test_identical_instants_preserve_three_original_spellings():
    originals = ("2026-09-12T12:00:00.120000Z", "2026-09-12T08:00:00.12-04:00",
                 "2026-09-12T17:30:00,120+05:30")
    result = compare_capture_clocks({"sample-A": originals[0]}, {"sample-A": originals[1]},
                                    {"sample-A": originals[2]}, max_skew_seconds=0)
    assert result["common_samples"] == 1 and result["maximum_observed_skew_seconds"] == 0
    assert result["exceeded_sample_ids"] == []
    assert result["samples"] == [{
        "sample_id": "sample-A",
        "original_timestamps": dict(zip(("camera", "gauge", "reference"), originals)),
        "maximum_skew_seconds": 0.0, "within_declared_limit": True,
    }]
    assert result["missing_ids_by_source"] == {"camera": [], "gauge": [], "reference": []}
    json.dumps(result, allow_nan=False)


def test_union_preserves_ids_and_insertion_order_including_missing_values():
    time = "2026-09-12T12:00:00Z"
    camera = {"z": time, "a": None}
    gauge = {"b": time, "z": time}
    reference = {"a": time, "c": time}
    result = compare_capture_clocks(camera, gauge, reference, max_skew_seconds=.1)
    assert [sample["sample_id"] for sample in result["samples"]] == ["z", "a", "b", "c"]
    assert result["common_samples"] == 0
    assert result["missing_ids_by_source"] == {"camera": ["a", "b", "c"],
                                               "gauge": ["a", "c"], "reference": ["z", "b"]}
    assert result["maximum_observed_skew_seconds"] is None
    assert all(sample["maximum_skew_seconds"] is None and sample["within_declared_limit"] is None
               for sample in result["samples"])
    assert result["samples"][0]["original_timestamps"] == {"camera": time, "gauge": time, "reference": None}
    assert camera == {"z": time, "a": None} and gauge == {"b": time, "z": time}
    json.dumps(result, allow_nan=False)


def test_maximum_pair_skew_uses_extreme_clocks_not_just_camera_pairs():
    result = compare_capture_clocks(
        {"s": "2026-09-12T12:00:00.500000Z"},
        {"s": "2026-09-12T12:00:00.400000Z"},
        {"s": "2026-09-12T12:00:00.600000Z"}, max_skew_seconds=.15)
    assert result["samples"][0]["maximum_skew_seconds"] == .2
    assert result["samples"][0]["within_declared_limit"] is False
    assert result["maximum_observed_skew_seconds"] == .2
    assert result["exceeded_sample_ids"] == ["s"]


def test_microsecond_boundary_is_not_rounded_or_given_an_extra_tolerance():
    zero = "2026-09-12T12:00:00.000000Z"
    camera = {"at": zero, "over": zero, "exact": zero}
    gauge = {"at": "2026-09-12T12:00:00.000001Z", "over": "2026-09-12T12:00:00.000002Z", "exact": zero}
    reference = dict(camera)
    result = compare_capture_clocks(camera, gauge, reference, max_skew_seconds=1e-6)
    assert [sample["within_declared_limit"] for sample in result["samples"]] == [True, False, True]
    assert result["exceeded_sample_ids"] == ["over"]
    assert result["maximum_observed_skew_seconds"] == 2e-6


def test_submicrosecond_limit_is_not_rounded_up_to_one_microsecond():
    start = "2026-09-12T12:00:00.000000Z"
    end = "2026-09-12T12:00:00.000001Z"
    result = compare_capture_clocks({"s": start}, {"s": end}, {"s": start},
                                    max_skew_seconds=.999999e-6)
    assert result["samples"][0]["within_declared_limit"] is False


def test_microsecond_above_a_large_limit_is_not_lost_in_float_seconds_conversion():
    start = datetime(2026, 9, 12, tzinfo=timezone.utc)
    end = start + timedelta(seconds=100_000_000_000, microseconds=1)
    assert (end-start).total_seconds() == 100_000_000_000  # float seconds lose the microsecond
    result = compare_capture_clocks({"s": start.isoformat()}, {"s": end.isoformat()},
                                    {"s": start.isoformat()}, max_skew_seconds=100_000_000_000)
    assert result["samples"][0]["within_declared_limit"] is False
    assert result["exceeded_sample_ids"] == ["s"]


def test_equal_timestamps_on_different_ids_are_not_repaired_into_pairs():
    time = "2026-09-12T12:00:00Z"
    result = compare_capture_clocks({"camera-id": time}, {"gauge-id": time}, {"ref-id": time}, max_skew_seconds=0)
    assert result["common_samples"] == 0
    assert len(result["samples"]) == 3
    assert "not true simultaneity" in result["note"]


def test_empty_sources_have_no_invented_observed_skew():
    result = compare_capture_clocks({}, {}, {}, max_skew_seconds=0)
    assert result["samples"] == [] and result["common_samples"] == 0
    assert result["maximum_observed_skew_seconds"] is None
    assert result["exceeded_sample_ids"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("limit", [True, np.bool_(False), -1, np.nan, np.inf, -np.inf, "0.1", None, 1j])
def test_limit_must_be_declared_as_finite_nonnegative_numeric_nonboolean(limit):
    with pytest.raises(ValueError, match="max_skew_seconds"):
        compare_capture_clocks({}, {}, {}, max_skew_seconds=limit)


def test_limit_has_no_default():
    with pytest.raises(TypeError):
        compare_capture_clocks({}, {}, {})


@pytest.mark.parametrize("source", [None, [], "timestamps", {"": None}, {"   ": None}, {3: None}, {True: None}])
@pytest.mark.parametrize("name", ["camera", "gauge", "reference"])
def test_sources_and_sample_ids_are_validated(source, name):
    sources = {"camera": {}, "gauge": {}, "reference": {}}
    sources[name] = source
    with pytest.raises(ValueError, match=name):
        compare_capture_clocks(**sources, max_skew_seconds=0)


def test_invalid_present_timestamp_is_not_hidden_by_missing_other_sources():
    with pytest.raises(ValueError, match="camera timestamp for sample 'bad'"):
        compare_capture_clocks({"bad": "2026-09-12T12:00:00"}, {}, {}, max_skew_seconds=0)
