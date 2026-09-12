"""Frame metadata policy and analytic, unit-aware signal checks."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from set_lcm.frame_quality import audit_frames, frame_content_hash, frame_spectrum


def _audit(n=8, period=.1, **overrides):
    args = dict(frame_ids=range(n), capture_times=np.arange(n) * period,
                content_hashes=[f"hash-{i}" for i in range(n)], time_origin="camera trigger epoch",
                source_kind="synthetic_test", expected_period=period)
    args.update(overrides)
    return audit_frames(**args)


def test_regular_decimal_clock_admits_without_rounding_or_changing_metadata():
    times = np.arange(8) * .1 + 1.2
    audit = _audit(capture_times=times, source_indices=range(40, 48), frame_ids=range(100, 108))
    assert audit.admitted == (True,) * 8
    assert audit.flags == ((),) * 8 and audit.spectral_eligible
    assert audit.capture_times == tuple(times)
    assert audit.source_indices == tuple(range(40, 48))
    assert audit.frame_ids == tuple(range(100, 108))
    assert audit.source_kinds == ("synthetic_test",) * 8


def test_duplicate_bytes_are_advisory_for_a_static_scene_unless_policy_rejects():
    hashes = [frame_content_hash(np.full((4, 6), 12, dtype=np.uint8))] * 8
    audit = _audit(content_hashes=hashes)
    assert audit.admitted == (True,) * 8
    assert audit.flags[0] == ()
    assert all(flags == ("repeated_content",) for flags in audit.flags[1:])
    assert audit.spectral_eligible
    result = frame_spectrum(np.full(8, 12.0), audit, feature_unit="brightness count")
    assert result.peak_frequency_hz is None and result.detrended_rms == 0
    assert np.all(np.array(result.power_density) == 0)
    strict = _audit(content_hashes=hashes, reject_repeated_content=True)
    assert strict.admitted == (True,) + (False,) * 7
    assert not strict.spectral_eligible
    with pytest.raises(ValueError, match="admitted"):
        frame_spectrum(np.ones(8), strict, feature_unit="count")


def test_nonadjacent_repeated_content_is_still_visible_without_false_fault_claim():
    audit = _audit(content_hashes=["a", "b", "c", "a", "e", "f", "g", "h"])
    assert "repeated_content" in audit.flags[3] and audit.admitted[3]
    assert any("static scene" in assumption for assumption in audit.assumptions)


@pytest.mark.parametrize("field,values,flag", [
    ("frame_ids", [0, 1, 1, 3], "duplicate_frame_id"),
    ("capture_times", [0.0, .1, .1, .3], "duplicate_timestamp"),
    ("frame_ids", [0, 2, 1, 3], "backward_frame_id"),
    ("capture_times", [0.0, .2, .1, .3], "backward_timestamp"),
])
def test_duplicate_and_backward_rows_are_preserved_but_refused(field, values, flag):
    audit = _audit(4, **{field: values})
    assert flag in audit.flags[2] and not audit.admitted[2]
    assert tuple(getattr(audit, field)) == tuple(values)
    assert not audit.spectral_eligible


def test_a_backward_clock_cannot_recover_admission_before_reaching_the_high_watermark():
    audit = _audit(4, period=1, capture_times=[3, 1, 2, 4], frame_ids=[3, 1, 2, 4])
    assert "backward_timestamp" in audit.flags[2]
    assert "backward_frame_id" in audit.flags[2]
    assert not audit.admitted[2]


def test_gap_is_explicit_and_never_silently_interpolated():
    audit = _audit(4, frame_ids=[0, 1, 3, 4], capture_times=[0, .1, .3, .4],
                   source_indices=[10, 11, 13, 14])
    assert set(audit.flags[2]) == {"frame_id_gap", "source_index_gap"}
    assert len(audit.capture_times) == 4 and audit.capture_times[2] == .3
    assert all(audit.admitted)
    with pytest.raises(ValueError, match="contiguous"):
        frame_spectrum(np.ones(4), audit, feature_unit="px")


def test_compressed_clock_cannot_hide_a_capture_counter_gap():
    audit = _audit(4, frame_ids=[0, 1, 3, 4])
    assert not audit.admitted[2]
    assert set(audit.flags[2]) == {"frame_id_gap", "clock_outside_tolerance", "short_interval"}
    assert not audit.spectral_eligible


def test_original_row_gap_blocks_fft_without_changing_physical_clock_admission():
    audit = _audit(4, source_indices=[0, 1, 3, 4])
    assert all(audit.admitted)
    assert audit.flags[2] == ("source_index_gap",)
    with pytest.raises(ValueError, match="contiguous"):
        frame_spectrum(np.ones(4), audit, feature_unit="px")


def test_external_missing_marker_retains_known_timeline_and_hash_absence():
    audit = _audit(4, content_hashes=["a", None, "c", "d"], missing=[False, True, False, False])
    assert audit.flags[1] == ("missing",)
    assert audit.content_hashes[1] is None and audit.capture_times[1] == .1
    assert not audit.admitted[1] and not audit.spectral_eligible
    with pytest.raises(ValueError, match="admitted"):
        frame_spectrum(np.zeros(4), audit, feature_unit="px")


def test_generated_views_cannot_be_admitted_as_independent_measurements():
    audit = _audit(4, source_kind=["measured", "generated", "synthetic_test", "measured"])
    assert audit.flags[1] == ("generated_source",) and not audit.admitted[1]
    assert audit.admitted[0] and audit.admitted[2]
    forged = replace(audit, admitted=(True,) * 4, spectral_eligible=True)
    with pytest.raises(ValueError, match="generated"):
        frame_spectrum(np.ones(4), forged, feature_unit="px")


def test_audit_jitter_admission_does_not_authorize_fft_resampling():
    times = np.arange(8, dtype=float)
    times[3] += .05
    audit = _audit(period=1, capture_times=times, max_jitter=.15)
    assert all(audit.admitted) and audit.capture_times == tuple(times)
    assert not audit.spectral_eligible
    with pytest.raises(ValueError, match="uniform capture clock"):
        frame_spectrum(np.arange(8), audit, feature_unit="px")
    strict = _audit(period=1, capture_times=times, max_jitter=.01)
    assert "long_interval" in strict.flags[3]
    assert "short_interval" in strict.flags[4]
    assert not strict.admitted[3] and not strict.admitted[4]


@pytest.mark.parametrize("n,k", [(64, 7), (63, 5)])
def test_sinusoid_frequency_and_integrated_psd_match_analytic_power(n, k):
    period, amplitude = .02, 3.0
    values = amplitude * np.sin(2 * np.pi * k * np.arange(n) / n)
    audit = _audit(n, period, source_indices=range(20, 20+n), frame_ids=range(90, 90+n))
    result = frame_spectrum(values, audit, feature_unit="px", detrend="none", window="boxcar")
    assert result.peak_frequency_hz == pytest.approx(k / (n*period))
    assert result.detrended_rms == pytest.approx(amplitude/np.sqrt(2))
    df = 1 / (n*period)
    assert sum(result.power_density) * df == pytest.approx(amplitude**2 / 2, rel=2e-13)
    assert result.power_density[k] == pytest.approx(amplitude**2 / (2*df), rel=2e-13)
    assert result.source_indices == tuple(range(20, 20+n))
    assert result.frame_ids == tuple(range(90, 90+n))
    assert result.capture_times == audit.capture_times


def test_dc_and_nyquist_bins_are_not_doubled():
    n, period = 16, .25
    values = 2 + 3 * (-1.0)**np.arange(n)
    result = frame_spectrum(values, _audit(n, period), feature_unit="count", detrend="none", window="boxcar")
    df = 1/(n*period)
    assert result.power_density[0] * df == pytest.approx(4)
    assert result.power_density[-1] * df == pytest.approx(9)
    assert sum(result.power_density) * df == pytest.approx(13)
    assert result.peak_frequency_hz == 1/(2*period)


def test_hann_psd_integral_is_window_weighted_mean_square():
    n, period = 64, .1
    j = np.arange(n)
    values = 1.5*np.sin(2*np.pi*8*j/n) + .4*np.cos(2*np.pi*3*j/n)
    result = frame_spectrum(values, _audit(n, period), feature_unit="px", detrend="none", window="hann")
    weights = .5 * (1 - np.cos(2*np.pi*j/(n-1)))
    expected = np.sum(weights**2 * values**2)/np.sum(weights**2)
    assert sum(result.power_density)/(n*period) == pytest.approx(expected, rel=3e-13)
    assert result.peak_frequency_hz == pytest.approx(8/(n*period))


def test_linear_detrending_removes_affine_brightness_without_claiming_a_fault():
    values = 6 + np.arange(32)*.5
    result = frame_spectrum(values, _audit(32), feature_unit="count", detrend="linear")
    assert result.detrended_rms == 0 and result.peak_frequency_hz is None
    assert any("not calibrated" in assumption for assumption in result.assumptions)
    assert result.values == tuple(values)


def test_time_origin_and_feature_units_change_only_the_corresponding_outputs():
    values = np.sin(2*np.pi*np.arange(32)/8)
    a = frame_spectrum(values, _audit(32), feature_unit="m", detrend="none")
    shifted = _audit(32, capture_times=100 + np.arange(32)*.1, time_origin="other origin")
    b = frame_spectrum(1000*values, shifted, feature_unit="mm", detrend="none")
    np.testing.assert_array_equal(a.frequencies_hz, b.frequencies_hz)
    np.testing.assert_allclose(np.array(b.power_density)/1e6, a.power_density, rtol=1e-12, atol=1e-25)
    assert a.time_origin != b.time_origin and b.feature_unit == "mm"


def test_consistently_wrong_clock_requires_external_reference_not_a_claimed_detector():
    audit = _audit(8, period=100)  # Milliseconds mislabeled as seconds, including the declared period.
    assert all(audit.admitted)
    assert any("no clock units" in assumption for assumption in audit.assumptions)
    wrong_reference = _audit(8, period=.1, capture_times=np.arange(8)*100)
    assert not any(wrong_reference.admitted[1:])


def test_hash_is_exact_in_dtype_shape_and_pixels_but_not_memory_layout():
    frame = np.arange(12, dtype=np.uint8).reshape(3, 4)
    digest = frame_content_hash(frame)
    assert len(digest) == 64
    assert digest == frame_content_hash(np.asfortranarray(frame))
    assert digest != frame_content_hash(frame.reshape(2, 6))
    assert digest != frame_content_hash(frame.astype(np.uint16))
    changed = frame.copy()
    changed[0, 0] += 1
    assert digest != frame_content_hash(changed)
    integers = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert frame_content_hash(integers) != frame_content_hash(integers.astype(">u2"))


def test_results_are_frozen_defensive_and_strict_json_safe():
    times, hashes = np.arange(8)*.1, [f"h{i}" for i in range(8)]
    audit = _audit(capture_times=times, content_hashes=hashes)
    times[:] = -1
    hashes[0] = "modified"
    assert audit.capture_times[0] == 0 and audit.content_hashes[0] == "h0"
    result = frame_spectrum(np.ones(8), audit, feature_unit="px")
    for record in (audit, result):
        encoded = json.dumps(record.as_dict(), allow_nan=False)
        assert "NaN" not in encoded and "Infinity" not in encoded
        with pytest.raises(FrozenInstanceError):
            record.time_origin = "changed"
    assert json.loads(json.dumps(result.as_dict()))["peak_frequency_hz"] is None


@pytest.mark.parametrize("overrides", [
    {"frame_ids": [0, 1.0, 2, 3]}, {"frame_ids": [0, True, 2, 3]},
    {"frame_ids": [0, -1, 2, 3]}, {"capture_times": [0, .1, np.nan, .3]},
    {"capture_times": [0, .1, 1j, .3]}, {"capture_times": [[0, .1, .2, .3]]},
    {"content_hashes": ["a", None, "c", "d"]}, {"content_hashes": "abcd"},
    {"content_hashes": ["a", " ", "c", "d"]}, {"source_kind": "unknown"},
    {"source_kind": ["measured"]}, {"missing": [0, 1, 0, 0]},
    {"time_origin": ""}, {"expected_period": 0}, {"expected_period": True},
    {"expected_period": np.inf}, {"max_jitter": -.1}, {"max_jitter": .05},
    {"source_indices": [0, 2, 1, 3]}, {"source_indices": [0, 1, 1, 3]},
    {"reject_repeated_content": 1},
])
def test_malformed_frame_contracts_are_refused(overrides):
    with pytest.raises(ValueError):
        _audit(4, **overrides)


@pytest.mark.parametrize("kwargs", [
    {"values": [1, 2]}, {"values": [1, 2, 3, np.inf]}, {"values": [1, 2, 3, 4j]},
    {"feature_unit": ""}, {"detrend": "quadratic"}, {"window": "unknown"},
    {"uniform_tolerance": -.1}, {"uniform_tolerance": np.inf}, {"uniform_tolerance": .05},
])
def test_malformed_spectral_contracts_are_refused(kwargs):
    arguments = dict(values=np.arange(4), audit=_audit(4), feature_unit="px")
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        frame_spectrum(**arguments)


def test_short_windows_and_unrepresentable_spectral_power_are_refused():
    with pytest.raises(ValueError, match="four"):
        frame_spectrum(np.ones(3), _audit(3), feature_unit="px")
    with pytest.raises(ValueError, match="finite numeric range"):
        frame_spectrum(np.array([1e300, -1e300]*4), _audit(), feature_unit="px", detrend="none")


@pytest.mark.parametrize("frame", [
    np.array([]), np.ones((2, 2), dtype=complex), np.ones((2, 2), dtype=object),
    np.ones((0, 2)), np.ones((2, 2, 2, 2)), [[1, 2], [3, 4]],
])
def test_hash_rejects_ambiguous_nonframe_representations(frame):
    with pytest.raises(ValueError):
        frame_content_hash(frame)
