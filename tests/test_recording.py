"""Recording preflight contracts using temporary, explicitly mocked evidence.

No fixture is measured data: 'measured' is a declaration exercised by these
tests, and all media/covariance bytes announce that they are mocks. Acceptance
means ready for human review, never quantitative or physical validation.
"""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from set_lcm.recording import (
    FRAME_COLUMNS,
    LEVEL_COLUMNS,
    check_recording,
    initialize_recording,
    main,
    recording_plan,
)


TABLE_PATHS = {
    "camera": "observations/frames.csv",
    "gauge": "observations/gauge.csv",
    "reference": "evaluation/reference.csv",
}


def save_plan(folder, plan):
    (folder / "recording.json").write_text(json.dumps(plan, allow_nan=False), encoding="utf-8")


def read_rows(folder, source):
    with (folder / TABLE_PATHS[source]).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(folder, source, rows):
    columns = FRAME_COLUMNS if source == "camera" else LEVEL_COLUMNS
    with (folder / TABLE_PATHS[source]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def edit_row(folder, source, index, **changes):
    rows = read_rows(folder, source)
    rows[index].update(changes)
    write_rows(folder, source, rows)


def assert_issue(report, status, field, message=None):
    assert report["status"] == status
    matches = [issue for issue in report["issues"] if field in issue["field"]]
    assert matches, report
    if message is not None:
        assert any(message.lower() in issue["message"].lower() for issue in matches), matches


def utc(seconds):
    value = datetime(2026, 9, 12, 12, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


@pytest.fixture
def mocked_recording(tmp_path):
    """Schema fixture only: mock bytes are deliberately not decodable media."""
    folder = tmp_path / "TEST-ONLY-mocked-recording"
    initialize_recording(folder)

    def mock_asset(relative):
        content = ("UNIT TEST MOCK, NOT ACQUIRED DATA: " + relative + "\n").encode()
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}

    plan = recording_plan()
    plan.update(
        recording_id="TEST-ONLY-held-out-metadata",
        source_kind="measured",  # Declaration under test; this fixture is not a real recording.
        measurand="TEST ONLY: mocked level metadata; no measured quantity is represented",
        level_datum="TEST ONLY: arbitrary metadata datum, not surveyed",
        fixture_notice="SYNTHETIC TEST METADATA AND MOCK FILE BYTES; NOT REAL MEASUREMENTS",
        frame_counter_kind="hardware_capture",
        frame_processing_evidence=mock_asset("evidence/frame-processing.mock"),
    )
    for name, spec in plan["instruments"].items():
        spec.update(id=f"TEST-ONLY-{name}", timestamp_meaning="declared exposure midpoint",
                    observation_support="explicit timestamp support, mocked for unit tests",
                    instrument_evidence=mock_asset(f"evidence/{name}.mock"))
    plan["clock"].update(max_pair_skew_seconds=0.025,
                         synchronization_evidence=mock_asset("evidence/clock.mock"))
    plan["calibration"].update(
        session_ids=["TEST-ONLY-calibration-session"],
        method="MOCK: no calibration was performed",
        model_evidence=mock_asset("calibration/model.mock"),
        parameter_covariance_evidence=mock_asset("calibration/parameters.mock"),
    )
    plan["development_session_ids"] = ["TEST-ONLY-development-session"]
    plan["locked_configuration_evidence"] = mock_asset("development/locked.mock")
    plan["uncertainty"].update(
        joint_covariance_evidence=mock_asset("evidence/covariance.mock"),
        reference_dependence="MOCK dependency statement, not an independence finding",
        reference_dependence_evidence=mock_asset("evidence/reference-dependence.mock"),
    )
    plan["acceptance"].update(maximum_reference_discrepancy_m=0.123,
                              criterion_evidence=mock_asset("development/criterion.mock"))
    plan["raw_videos"] = [{"id": "TEST-ONLY-not-a-video", **mock_asset("raw/video.mock")}]
    save_plan(folder, plan)
    frames = []
    for index in range(2):
        payload = mock_asset(f"observations/frame-{index}.mock")
        frames.append({
            "sample_id": f"TEST-event-{index}", "frame_id": str(10 + index),
            "capture_utc": utc(index), "exposure_start_utc": utc(index - 0.001),
            "exposure_end_utc": utc(index + 0.001), "present": "true",
            "frame_path": payload["path"], "frame_sha256": payload["sha256"],
            "source_video_id": "TEST-ONLY-not-a-video", "source_frame_index": str(20 + index),
        })
    write_rows(folder, "camera", frames)
    for source, skew in (("gauge", 0.01), ("reference", 0.02)):
        write_rows(folder, source, [{
            "sample_id": f"TEST-event-{index}", "capture_utc": utc(index + skew),
            "support_start_utc": utc(index + skew), "support_end_utc": utc(index + skew),
            "level_m": str(0.4 + index * 0.1), "standard_uncertainty_m": "0.01",
        } for index in range(2)])
    return folder, plan


def test_plan_has_unknown_equipment_values_and_independent_nested_objects():
    plan = recording_plan()
    assert plan["recording_id"] is None and plan["source_kind"] is None
    assert plan["measurand"] is None and plan["level_datum"] is None
    assert plan["clock"]["max_pair_skew_seconds"] is None
    assert plan["acceptance"]["maximum_reference_discrepancy_m"] is None
    assert plan["calibration"]["session_ids"] == []
    assert plan["development_session_ids"] == [] and plan["raw_videos"] == []
    assert all(spec["id"] is None for spec in plan["instruments"].values())
    plan["instruments"]["camera"]["instrument_evidence"]["path"] = "changed"
    assert plan["instruments"]["gauge"]["instrument_evidence"]["path"] is None
    assert recording_plan()["instruments"]["camera"]["instrument_evidence"]["path"] is None


def test_initialization_writes_headers_and_no_mock_or_real_measurements(tmp_path):
    folder = tmp_path / "blank"
    manifest = initialize_recording(folder)
    assert manifest == folder / "recording.json"
    assert json.loads(manifest.read_text()) == recording_plan()
    assert all(read_rows(folder, source) == [] for source in TABLE_PATHS)
    assert list((folder / "raw").iterdir()) == []
    with (folder / "evaluation/events.csv").open(newline="") as handle:
        assert len(list(csv.reader(handle))) == 1
    report = check_recording(folder)
    assert report["status"] == "incomplete"
    assert report["rows"] == {"camera": 0, "gauge": 0, "reference": 0}
    assert report["verified_assets"] == []
    assert report["quantitative_evaluation_performed"] is False
    assert report["field_validation_performed"] is False
    json.dumps(report, allow_nan=False)


def test_default_cli_summary_does_not_claim_completed_validation(tmp_path, capsys):
    initialize_recording(tmp_path / "blank")
    assert main(["check", str(tmp_path / "blank")]) == 2
    summary = capsys.readouterr().out
    assert "Recording preflight: incomplete" in summary
    assert "No field validation or quantitative evaluation has been performed" in summary
    assert "--json" in summary


@pytest.mark.parametrize("existing_kind", ["file", "nonempty_directory"])
def test_initialization_never_overwrites_existing_evidence(tmp_path, existing_kind):
    destination = tmp_path / "existing"
    if existing_kind == "file":
        destination.write_bytes(b"original evidence")
        sentinel = destination
    else:
        destination.mkdir()
        sentinel = destination / "original.bin"
        sentinel.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="never overwritten"):
        initialize_recording(destination)
    assert sentinel.read_bytes() == b"original evidence"


def test_cli_blank_check_returns_two_and_writes_only_external_report(tmp_path, capsys):
    folder = tmp_path / "blank"
    assert main(["init", str(folder)]) == 0
    capsys.readouterr()
    report_path = tmp_path / "review" / "preflight.json"
    assert main(["check", str(folder), "--out-report", str(report_path), "--json"]) == 2
    actual = json.loads(capsys.readouterr().out)
    assert actual["status"] == "incomplete"
    assert json.loads(report_path.read_text()) == actual


def test_cli_report_cannot_overwrite_input_evidence(mocked_recording):
    folder, _ = mocked_recording
    path = folder / "recording.json"
    original = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["check", str(folder), "--out-report", str(path)])
    assert exc.value.code == 2
    assert path.read_bytes() == original


def test_mocked_declarations_only_reach_ready_for_review(mocked_recording, capsys):
    folder, _ = mocked_recording
    before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    report = check_recording(folder)
    assert report["status"] == "ready_for_review", report["issues"]
    assert report["field_validation_performed"] is False
    assert report["quantitative_evaluation_performed"] is False
    assert report["issues"] == []
    assert report["rows"] == {"camera": 2, "gauge": 2, "reference": 2}
    assert report["synchronization"]["common_samples"] == 2
    assert report["synchronization"]["maximum_observed_skew_seconds"] == pytest.approx(0.02)
    assert report["synchronization"]["samples"][0]["original_timestamps"]["reference"] == utc(0.02)
    assert any("not numerically interpreted" in text for text in report["limitations"])
    assert before == {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    assert main(["check", str(folder), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready_for_review"


@pytest.mark.parametrize("payload", [
    "not JSON", "[]", '{"schema":"fsre-tank-recording-v1","schema":"duplicate"}',
    '{"schema":"fsre-tank-recording-v1","clock":{"x":1,"x":2}}',
    '{"schema":"fsre-tank-recording-v1","unused":NaN}',
    '{"schema":"fsre-tank-recording-v1","unused":Infinity}',
    '{"schema":"fsre-tank-recording-v1","unused":-Infinity}',
])
def test_invalid_or_ambiguous_json_is_refused(tmp_path, payload):
    (tmp_path / "recording.json").write_text(payload, encoding="utf-8")
    assert check_recording(tmp_path)["status"] == "invalid"


@pytest.mark.parametrize("unsafe", ["../outside.mock", "/absolute.mock", "C:/outside.mock", "raw\\video.mock", "//host/share.mock"])
def test_asset_paths_cannot_escape_recording(mocked_recording, unsafe):
    folder, plan = mocked_recording
    plan["raw_videos"][0]["path"] = unsafe
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "raw_videos[0]", "path")


def test_symlink_resolving_outside_recording_is_refused(mocked_recording, tmp_path):
    folder, plan = mocked_recording
    outside = tmp_path / "outside.mock"
    outside.write_bytes(b"outside test bytes")
    link = folder / "raw" / "linked.mock"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symbolic links is unavailable on this runner")
    plan["raw_videos"][0].update(path="raw/linked.mock", sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "raw_videos[0]", "outside")


def test_manifest_symlink_cannot_move_declarations_outside_recording(mocked_recording, tmp_path):
    folder, _ = mocked_recording
    manifest = folder / "recording.json"
    outside = tmp_path / "outside-recording.json"
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    try:
        manifest.symlink_to(outside)
    except OSError:
        pytest.skip("creating symbolic links is unavailable on this runner")
    assert_issue(check_recording(folder), "invalid", "recording.json", "outside")


def test_asset_checksum_mismatch_is_not_a_successful_review(mocked_recording):
    folder, plan = mocked_recording
    (folder / plan["calibration"]["model_evidence"]["path"]).write_bytes(b"changed mock bytes")
    assert_issue(check_recording(folder), "invalid", "calibration.model_evidence", "checksum")


@pytest.mark.parametrize("digest", ["x" * 64, "0" * 63, "A" * 64])
def test_malformed_digest_is_refused(mocked_recording, digest):
    folder, plan = mocked_recording
    plan["raw_videos"][0]["sha256"] = digest
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "raw_videos[0]", "SHA-256")


@pytest.mark.parametrize("field", ["joint_covariance_evidence", "reference_dependence_evidence"])
def test_missing_uncertainty_evidence_stays_incomplete(mocked_recording, field):
    folder, plan = mocked_recording
    plan["uncertainty"][field] = {"path": None, "sha256": None}
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "incomplete", "uncertainty." + field)


@pytest.mark.parametrize("source", ["gauge", "reference"])
@pytest.mark.parametrize("uncertainty,status", [("", "incomplete"), ("-0.1", "invalid"), ("nan", "invalid"), ("inf", "invalid")])
def test_missing_or_invalid_standard_uncertainty_is_not_zero(mocked_recording, source, uncertainty, status):
    folder, _ = mocked_recording
    edit_row(folder, source, 0, standard_uncertainty_m=uncertainty)
    assert_issue(check_recording(folder), status, "standard_uncertainty_m")


@pytest.mark.parametrize("source_kind", ["generated", "synthetic_test", "unknown"])
def test_generated_or_synthetic_declaration_is_not_real_acquisition(mocked_recording, source_kind):
    folder, plan = mocked_recording
    plan["source_kind"] = source_kind
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "source_kind", "acquired")


@pytest.mark.parametrize("unit", ["cm", "ft", None, ""])
def test_wrong_or_missing_units_are_not_silently_metres(mocked_recording, unit):
    folder, plan = mocked_recording
    plan["level_unit"] = unit
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "level_unit", "metres")


def test_reference_cannot_reuse_gauge_instrument_identity(mocked_recording):
    folder, plan = mocked_recording
    plan["instruments"]["reference"]["id"] = plan["instruments"]["gauge"]["id"]
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "instruments", "distinct")


@pytest.mark.parametrize("borrower,lender", [("reference", "gauge"), ("gauge", "camera"),
                                             ("camera", "reference")])
def test_distinct_instruments_cannot_share_one_evidence_document(mocked_recording, borrower, lender):
    """Distinct identities documented by one file are one document, whatever the identities say.

    This catches only the crudest dependence -- literally the same bytes. Real dependence
    between a gauge and the reference that evaluates it is physical and invisible to a
    preflight; uncertainty.reference_dependence is where that gets declared.
    """
    folder, plan = mocked_recording
    plan["instruments"][borrower]["instrument_evidence"] = plan["instruments"][lender]["instrument_evidence"]
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "instruments", "distinct instrument evidence")


@pytest.mark.parametrize("source", ["gauge", "reference"])
def test_a_declared_zero_uncertainty_is_surfaced_rather_than_accepted_in_silence(mocked_recording, source):
    """Zero is the strongest uncertainty claim available and the kit should not pass it quietly.

    It is not refused: a source may genuinely state it, and elsewhere this repository carries
    NOAA's stated 0.000 as R = 0 by design. It is raised for review, as a decoder-numbered
    frame counter is, so a reviewer sees the claim instead of inheriting it.
    """
    folder, _ = mocked_recording
    edit_row(folder, source, 0, standard_uncertainty_m="0")
    report = check_recording(folder)
    assert report["status"] == "ready_for_review"
    matches = [issue for issue in report["issues"]
               if issue["kind"] == "review" and "standard_uncertainty_m" in issue["field"]]
    assert matches, report["issues"]
    assert "declares this reading exact" in matches[0]["message"]


def test_an_ordinary_positive_uncertainty_raises_nothing(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "gauge", 0, standard_uncertainty_m="0.004")
    report = check_recording(folder)
    assert not [issue for issue in report["issues"] if "standard_uncertainty_m" in issue["field"]]


@pytest.mark.parametrize("split_problem", ["calibration_is_heldout", "development_is_heldout", "duplicate", "overlap"])
def test_whole_session_split_must_be_disjoint(mocked_recording, split_problem):
    folder, plan = mocked_recording
    if split_problem == "calibration_is_heldout":
        plan["calibration"]["session_ids"] = [plan["recording_id"]]
    elif split_problem == "development_is_heldout":
        plan["development_session_ids"] = [plan["recording_id"]]
    elif split_problem == "duplicate":
        plan["calibration"]["session_ids"] *= 2
    else:
        plan["development_session_ids"] = plan["calibration"]["session_ids"][:]
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "session_ids")


@pytest.mark.parametrize("source", ["camera", "gauge", "reference"])
@pytest.mark.parametrize("timestamp", ["2026-09-12T12:00:00", "2026-09-12", "2026-09-12T12:00:00.1234567Z"])
def test_naive_or_precision_losing_timestamps_are_refused(mocked_recording, source, timestamp):
    folder, _ = mocked_recording
    edit_row(folder, source, 0, capture_utc=timestamp)
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS[source] + ":2.time")


@pytest.mark.parametrize("source", ["camera", "gauge", "reference"])
def test_capture_must_be_inside_temporal_support(mocked_recording, source):
    folder, _ = mocked_recording
    key = "exposure_start_utc" if source == "camera" else "support_start_utc"
    edit_row(folder, source, 0, **{key: utc(0.5)})
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS[source] + ":2.time", "support")


@pytest.mark.parametrize("source", ["camera", "gauge", "reference"])
def test_duplicate_sample_ids_are_not_implicitly_combined(mocked_recording, source):
    folder, _ = mocked_recording
    edit_row(folder, source, 1, sample_id="TEST-event-0")
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS[source], "Duplicate sample")


def test_source_frame_order_is_checked_separately_from_capture_counter(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 1, source_frame_index="20")
    assert_issue(check_recording(folder), "invalid", "observations/frames.csv", "source")


@pytest.mark.parametrize("counter_kind", ["decoder_index", "unavailable"])
def test_nonhardware_counter_is_explicitly_reviewed_without_inventing_ids(mocked_recording, counter_kind):
    folder, plan = mocked_recording
    plan["frame_counter_kind"] = counter_kind
    save_plan(folder, plan)
    if counter_kind == "unavailable":
        rows = read_rows(folder, "camera")
        for row in rows:
            row["frame_id"] = ""
        write_rows(folder, "camera", rows)
    report = check_recording(folder)
    assert report["status"] == "ready_for_review"
    assert any(issue["kind"] == "review" and issue["field"] == "frame_counter_kind" for issue in report["issues"])
    assert report["synchronization"]["common_samples"] == 2
    if counter_kind == "unavailable":
        assert all(row["frame_id"] == "" for row in read_rows(folder, "camera"))


def test_unknown_counter_kind_and_missing_processing_evidence_are_reported(mocked_recording):
    folder, plan = mocked_recording
    plan["frame_counter_kind"] = "assume_acquisition_order"
    plan["frame_processing_evidence"] = {"path": None, "sha256": None}
    save_plan(folder, plan)
    report = check_recording(folder)
    assert_issue(report, "invalid", "frame_counter_kind")
    assert any(issue["kind"] == "missing" and issue["field"] == "frame_processing_evidence" for issue in report["issues"])


@pytest.mark.parametrize("field", ["frame_id", "source_frame_index"])
def test_unreasonable_integer_input_is_reported_instead_of_crashing(mocked_recording, field):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 0, **{field: "9" * 5000})
    assert_issue(check_recording(folder), "invalid", "observations/frames.csv")


def test_same_original_recording_cannot_be_reused_under_an_alias(mocked_recording):
    folder, plan = mocked_recording
    plan["raw_videos"].append({**plan["raw_videos"][0], "id": "TEST-ONLY-video-alias"})
    save_plan(folder, plan)
    edit_row(folder, "camera", 1, source_video_id="TEST-ONLY-video-alias", source_frame_index="20")
    # Relabeling the same file does not make its same decoded frame new evidence.
    assert check_recording(folder)["status"] == "invalid"


def test_copied_original_bytes_do_not_become_a_distinct_recording(mocked_recording):
    folder, plan = mocked_recording
    original = plan["raw_videos"][0]
    duplicate = folder / "raw/copied.mock"
    duplicate.write_bytes((folder / original["path"]).read_bytes())
    plan["raw_videos"].append({"id": "TEST-ONLY-copy", "path": "raw/copied.mock", "sha256": original["sha256"]})
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "raw_videos", "identical")


@pytest.mark.parametrize("counter_kind,expected_status", [("decoder_index", "ready_for_review"), ("hardware_capture", "invalid")])
def test_decoder_counter_can_restart_only_for_a_distinct_original(mocked_recording, counter_kind, expected_status):
    folder, plan = mocked_recording
    content = b"UNIT TEST MOCK: a distinct second original, not video data"
    (folder / "raw/second.mock").write_bytes(content)
    plan["raw_videos"].append({"id": "TEST-ONLY-second-original", "path": "raw/second.mock",
                               "sha256": hashlib.sha256(content).hexdigest()})
    plan["frame_counter_kind"] = counter_kind
    save_plan(folder, plan)
    edit_row(folder, "camera", 1, frame_id="0", source_video_id="TEST-ONLY-second-original", source_frame_index="0")
    report = check_recording(folder)
    assert report["status"] == expected_status, report["issues"]
    if counter_kind == "hardware_capture":
        assert_issue(report, "invalid", "observations/frames.csv", "counter")


def test_backward_frame_counter_is_refused(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 1, frame_id="9")
    assert_issue(check_recording(folder), "invalid", "observations/frames.csv", "counter")


def test_time_order_cannot_be_repaired_by_row_order_or_sample_ids(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "reference", 1, capture_utc=utc(-1), support_start_utc=utc(-1), support_end_utc=utc(-1))
    assert_issue(check_recording(folder), "invalid", "evaluation/reference.csv:3.time", "increase")


def test_skew_exceedance_preserves_original_times_without_alignment(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "reference", 0, capture_utc=utc(0.1), support_start_utc=utc(0.1), support_end_utc=utc(0.1))
    report = check_recording(folder)
    assert_issue(report, "invalid", "synchronization", "skew")
    assert report["synchronization"]["exceeded_sample_ids"] == ["TEST-event-0"]
    first = report["synchronization"]["samples"][0]
    assert first["original_timestamps"] == {"camera": utc(0), "gauge": utc(0.01), "reference": utc(0.1)}
    assert first["maximum_skew_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize("limit", [True, -1, "nan", "inf", "1e1000"])
def test_invalid_declared_skew_limit_is_not_guessed(mocked_recording, limit):
    folder, plan = mocked_recording
    plan["clock"]["max_pair_skew_seconds"] = limit
    save_plan(folder, plan)
    assert_issue(check_recording(folder), "invalid", "clock.max_pair_skew_seconds")


def test_declared_missing_payload_stays_missing_with_other_complete_samples(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 0, present="false", frame_path="", frame_sha256="", capture_utc="",
             exposure_start_utc="", exposure_end_utc="", source_video_id="", source_frame_index="")
    report = check_recording(folder)
    assert report["status"] == "ready_for_review"
    assert report["rows"]["camera"] == 2
    assert report["synchronization"]["common_samples"] == 1
    assert report["synchronization"]["missing_ids_by_source"]["camera"] == ["TEST-event-0"]
    assert report["synchronization"]["samples"][0]["maximum_skew_seconds"] is None


def test_missing_frame_cannot_contain_a_substitute_payload(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 0, present="false")
    assert_issue(check_recording(folder), "invalid", "observations/frames.csv", "substituted")


@pytest.mark.parametrize("source", ["camera", "gauge", "reference"])
def test_missing_observation_does_not_hide_malformed_provided_support(mocked_recording, source):
    folder, _ = mocked_recording
    if source == "camera":
        changes = dict(present="false", frame_path="", frame_sha256="", capture_utc="",
                       exposure_start_utc="not-a-time", exposure_end_utc="not-a-time")
    else:
        changes = dict(level_m="", standard_uncertainty_m="", capture_utc="",
                       support_start_utc="not-a-time", support_end_utc="not-a-time")
    edit_row(folder, source, 0, **changes)
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS[source])


@pytest.mark.parametrize("source", ["camera", "gauge", "reference"])
def test_missing_observation_can_retain_valid_partial_support_without_filling_capture(mocked_recording, source):
    folder, _ = mocked_recording
    if source == "camera":
        changes = dict(present="false", frame_path="", frame_sha256="", capture_utc="",
                       exposure_start_utc=utc(0), exposure_end_utc="")
    else:
        changes = dict(level_m="", standard_uncertainty_m="0.01", capture_utc="",
                       support_start_utc=utc(0), support_end_utc="")
    edit_row(folder, source, 0, **changes)
    report = check_recording(folder)
    assert report["status"] == "ready_for_review", report["issues"]
    assert report["synchronization"]["samples"][0]["original_timestamps"][source] is None
    assert read_rows(folder, source)[0]["capture_utc"] == ""


def test_missing_observation_does_not_hide_reversed_support(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "reference", 0, level_m="", standard_uncertainty_m="", capture_utc="",
             support_start_utc=utc(0.1), support_end_utc=utc(0))
    assert_issue(check_recording(folder), "invalid", "evaluation/reference.csv")


@pytest.mark.parametrize("source", ["gauge", "reference"])
def test_missing_level_does_not_hide_nonfinite_supplied_uncertainty(mocked_recording, source):
    folder, _ = mocked_recording
    edit_row(folder, source, 0, level_m="", standard_uncertainty_m="NaN", capture_utc="",
             support_start_utc="", support_end_utc="")
    assert_issue(check_recording(folder), "invalid", "standard_uncertainty_m")


def test_no_complete_capture_triple_is_incomplete(mocked_recording):
    folder, _ = mocked_recording
    rows = read_rows(folder, "reference")
    for row in rows:
        row.update(level_m="", standard_uncertainty_m="", capture_utc="", support_start_utc="", support_end_utc="")
    write_rows(folder, "reference", rows)
    report = check_recording(folder)
    assert_issue(report, "incomplete", "synchronization", "No complete")
    assert report["synchronization"]["common_samples"] == 0


def test_missing_present_frame_file_is_incomplete(mocked_recording):
    folder, _ = mocked_recording
    (folder / read_rows(folder, "camera")[0]["frame_path"]).unlink()
    assert_issue(check_recording(folder), "incomplete", "observations/frames.csv:2.frame", "absent")


def test_unknown_original_video_id_is_refused(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "camera", 0, source_video_id="not-in-manifest")
    assert_issue(check_recording(folder), "invalid", "observations/frames.csv", "original recording")


@pytest.mark.parametrize("csv_problem", ["wrong_header", "extra_column", "short_row"])
def test_malformed_csv_is_reported_without_inference(mocked_recording, csv_problem):
    folder, _ = mocked_recording
    path = folder / TABLE_PATHS["gauge"]
    text = path.read_text()
    if csv_problem == "wrong_header":
        text = text.replace("level_m", "level_cm", 1)
    elif csv_problem == "extra_column":
        lines = text.splitlines()
        lines[1] += ",extra"
        text = "\n".join(lines) + "\n"
    else:
        lines = text.splitlines()
        lines[1] = lines[1].rsplit(",", 1)[0]
        text = "\n".join(lines) + "\n"
    path.write_text(text)
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS["gauge"])


def test_oversized_csv_field_is_reported_instead_of_crashing(mocked_recording):
    folder, _ = mocked_recording
    edit_row(folder, "gauge", 0, sample_id="x" * 150000)
    assert_issue(check_recording(folder), "invalid", TABLE_PATHS["gauge"])


def test_preflight_never_calls_estimators_or_reads_intervention_labels(mocked_recording, monkeypatch):
    import set_lcm.camera as camera
    import set_lcm.camera_fusion as fusion
    import set_lcm.diagnostics as diagnostics
    import set_lcm.experiments.camera_baseline as baseline

    folder, _ = mocked_recording
    def forbidden(*args, **kwargs):
        raise AssertionError("an estimator must not receive the preflight's held-out evidence")
    monkeypatch.setattr(camera, "detect_level", forbidden)
    monkeypatch.setattr(fusion, "compare_level_sources", forbidden)
    monkeypatch.setattr(diagnostics, "diagnose", forbidden)
    monkeypatch.setattr(baseline, "infer_measurements", forbidden)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name != "events.csv", "intervention labels must not drive preflight admission"
        return original_open(path, *args, **kwargs)
    (folder / "evaluation/events.csv").write_bytes(b"intentionally unreadable scoring labels")
    monkeypatch.setattr(Path, "open", guarded_open)
    # A huge reference discrepancy still cannot be evaluated by metadata preflight.
    edit_row(folder, "reference", 0, level_m="123456789")
    report = check_recording(folder)
    assert report["status"] == "ready_for_review"
    assert report["quantitative_evaluation_performed"] is False
    assert report["field_validation_performed"] is False
