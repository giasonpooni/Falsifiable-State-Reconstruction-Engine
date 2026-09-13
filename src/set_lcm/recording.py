"""Prepare and inspect real tank-recording evidence without claiming field validation.

This preflight checks metadata, files, units and declared clocks. It does not decode
video, fit a real camera, establish synchronization or compute field accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

from .recording_clock import compare_capture_clocks, parse_capture_utc

SCHEMA = "fsre-tank-recording-v1"
FRAME_COLUMNS = ("sample_id", "frame_id", "capture_utc", "exposure_start_utc", "exposure_end_utc",
                 "present", "frame_path", "frame_sha256", "source_video_id", "source_frame_index")
LEVEL_COLUMNS = ("sample_id", "capture_utc", "support_start_utc", "support_end_utc",
                 "level_m", "standard_uncertainty_m")
TABLES = {"camera": ("observations/frames.csv", FRAME_COLUMNS),
          "gauge": ("observations/gauge.csv", LEVEL_COLUMNS),
          "reference": ("evaluation/reference.csv", LEVEL_COLUMNS)}


def recording_plan() -> dict:
    """An unrecorded plan: every equipment-dependent value remains explicitly unknown."""
    def asset():
        return {"path": None, "sha256": None}
    return {
        "schema": SCHEMA, "recording_id": None, "source_kind": None, "level_unit": "m",
        "measurand": None, "level_datum": None,
        "instruments": {name: {"id": None, "timestamp_meaning": None,
                                "observation_support": None, "instrument_evidence": asset()}
                        for name in ("camera", "gauge", "reference")},
        "frame_counter_kind": None, "frame_processing_evidence": asset(),
        "clock": {"max_pair_skew_seconds": None, "synchronization_evidence": asset()},
        "calibration": {"session_ids": [], "method": None, "model_evidence": asset(),
                        "parameter_covariance_evidence": asset()},
        "development_session_ids": [], "locked_configuration_evidence": asset(),
        "raw_videos": [],
        "uncertainty": {"joint_covariance_evidence": asset(),
                        "reference_dependence": None, "reference_dependence_evidence": asset()},
        "acceptance": {"maximum_reference_discrepancy_m": None, "criterion_evidence": asset()},
    }


def initialize_recording(directory: Path) -> Path:
    """Create blank planning files in a new/empty folder, never overwrite a recording."""
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("recording directory must be new or empty; existing evidence is never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "observations", "calibration", "development", "evaluation", "evidence"):
        (directory / name).mkdir()
    path = directory / "recording.json"
    path.write_text(json.dumps(recording_plan(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for relative, fields in TABLES.values():
        with (directory / relative).open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(fields)
    with (directory / "evaluation/events.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(("event_id", "start_utc", "end_utc", "event_type", "notes"))
    (directory / "README.md").write_text(
        "# Unrecorded tank experiment\n\n"
        "This folder contains a blank plan and CSV headers, not measured or simulated data.\n"
        "Complete recording.json with your equipment and evidence. Unknown values remain null.\n"
        "Use the repository's docs/TANK_RECORDING_PROTOCOL.md before recording.\n\n"
        "Keep original video in raw/; list each in raw_videos as {id, path, sha256}.\n"
        "Keep decoded frames in observations/, with their original video/index, file checksum,\n"
        "capture timestamp and exposure interval. Preserve missing rows: present=false, blank\n"
        "payload/time fields when unknown; do not invent a measured capture time.\n"
        "Declare frame_counter_kind as hardware_capture, decoder_index, or unavailable;\n"
        "decoder numbering cannot prove that the acquisition had no dropped frames.\n"
        "CSV sample_id links a declared acquisition event across instruments. It is not a\n"
        "new timestamp. Keep original device-clock mappings in synchronization evidence.\n"
        "Gauge and reference levels/standard uncertainties are in metres; preserve their\n"
        "actual temporal support. An instantaneous observation has equal support endpoints.\n\n"
        "Calibration and development sessions must differ from this held-out recording.\n"
        "Reference measurements and intervention events stay in evaluation/. Do not use them\n"
        "to tune camera extraction, synchronization, calibration or fusion on this recording.\n"
        "Scalar uncertainty columns do not replace full covariance evidence.\n\n"
        "Run: python -m set_lcm.recording check <this-folder>\n"
        "A passing preflight means ready_for_review, not validated equipment or measured accuracy.\n"
        "No compressed-video decoder or real-data inference adapter is supplied by this kit.\n",
        encoding="utf-8")
    return path


def _strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def check_recording(directory: Path) -> dict:
    """Inspect a held-out recording's declarations; never perform estimation or scoring."""
    root = Path(directory).resolve()
    issues, verified_assets = [], []
    tables = {}
    synchronization = None

    def issue(kind, field, message):
        issues.append({"kind": kind, "field": field, "message": message})

    def result():
        kinds = {item["kind"] for item in issues}
        return {"schema": SCHEMA, "status": "invalid" if "error" in kinds else
                "incomplete" if "missing" in kinds else "ready_for_review",
                "field_validation_performed": False, "quantitative_evaluation_performed": False,
                "issues": issues, "verified_assets": verified_assets,
                "rows": {name: len(rows) for name, rows in tables.items()},
                "synchronization": synchronization,
                "limitations": [
                    "Preflight checks declarations and byte consistency, not authenticity or physical accuracy.",
                    "Invalid/incomplete means this declared input contract needs attention, not that an instrument has failed.",
                    "UTC timestamps require capture-clock evidence; a small declared skew does not prove simultaneity.",
                    "Matched IDs do not align unequal clocks. Exposure/averaging support still needs a measurement model.",
                    "Covariance and calibration evidence files are retained for review, not numerically interpreted here.",
                    "Real image extraction, uncertain-anchor calibration and asynchronous fusion still require validation.",
                    "No held-out reference or intervention label is passed to an estimator by this tool."]}

    def object_field(value, field):
        if not isinstance(value, dict):
            issue("missing" if value is None else "error", field, "A declaration object is required.")
            return {}
        return value

    def text_field(value, field):
        if not isinstance(value, str) or not value.strip():
            issue("missing" if value is None or value == "" else "error", field, "A nonempty declaration is required.")
            return None
        return value

    def nonnegative(value, field):
        if value is None or value == "":
            issue("missing", field, "Declare a finite nonnegative value from evidence, not a guessed default.")
            return None
        if isinstance(value, bool):
            issue("error", field, "A boolean is not a measurement.")
            return None
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            number = math.nan
        if not math.isfinite(number) or number < 0:
            issue("error", field, "A finite nonnegative value is required.")
            return None
        return number

    def relative_path(value):
        if (not isinstance(value, str) or not value or "\\" in value
                or Path(value).is_absolute() or PureWindowsPath(value).drive
                or ".." in Path(value).parts):
            raise ValueError("Use a portable relative path inside the recording folder.")
        resolved = (root / value).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Evidence path resolves outside the recording folder.")
        return resolved

    def counter(value, field):
        if re.fullmatch(r"[0-9]+", value) is None:
            issue("error", field, "A nonnegative decimal frame counter is required.")
            return None
        try:
            return int(value)
        except ValueError:
            issue("error", field, "Frame counter exceeds the supported integer parser range.")
            return None

    def asset(value, field):
        value = object_field(value, field)
        path, digest = value.get("path"), value.get("sha256")
        if not path or not digest:
            issue("missing", field, "Supply a relative evidence path and its SHA-256 file checksum.")
            return
        try:
            resolved = relative_path(path)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("SHA-256 must contain 64 lowercase hexadecimal characters.")
            if not resolved.is_file():
                issue("missing", field, f"Evidence file is absent: {path}")
                return
            hasher = hashlib.sha256()
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != digest:
                raise ValueError("File checksum does not match its declaration.")
        except (OSError, ValueError) as error:
            issue("error", field, str(error))
            return
        verified_assets.append({"field": field, "path": path, "sha256": digest})
        return resolved, digest

    try:
        plan = _strict_json(relative_path("recording.json"))
    except (OSError, ValueError, RecursionError) as error:
        issue("missing" if isinstance(error, FileNotFoundError) else "error", "recording.json", str(error))
        return result()
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        issue("error", "schema", "Unsupported recording-plan schema.")
        return result()
    recording_id = text_field(plan.get("recording_id"), "recording_id")
    for field in ("measurand", "level_datum"):
        text_field(plan.get(field), field)
    source = text_field(plan.get("source_kind"), "source_kind")
    if source is not None and source != "measured":
        issue("error", "source_kind", "This real-recording preflight requires acquired measurements, not generated or synthetic substitutes.")
    if plan.get("level_unit") != "m":
        issue("error", "level_unit", "Level and standard-uncertainty CSV fields must be in metres.")
    counter_kind = text_field(plan.get("frame_counter_kind"), "frame_counter_kind")
    if counter_kind is not None and counter_kind not in ("hardware_capture", "decoder_index", "unavailable"):
        issue("error", "frame_counter_kind", "Declare hardware_capture, decoder_index, or unavailable.")
    if counter_kind in ("decoder_index", "unavailable"):
        issue("review", "frame_counter_kind", "Acquisition gaps cannot be established from a missing hardware counter or decoder numbering alone.")
    asset(plan.get("frame_processing_evidence"), "frame_processing_evidence")
    instruments = object_field(plan.get("instruments"), "instruments")
    instrument_ids = []
    for name in TABLES:
        spec = object_field(instruments.get(name), f"instruments.{name}")
        instrument_ids.append(text_field(spec.get("id"), f"instruments.{name}.id"))
        for field in ("timestamp_meaning", "observation_support"):
            text_field(spec.get(field), f"instruments.{name}.{field}")
        asset(spec.get("instrument_evidence"), f"instruments.{name}.instrument_evidence")
    if any(identifier is not None and instrument_ids.count(identifier) > 1 for identifier in instrument_ids):
        issue("error", "instruments", "Camera, gauge and evaluation reference must have distinct instrument identities.")
    clock = object_field(plan.get("clock"), "clock")
    skew_limit = nonnegative(clock.get("max_pair_skew_seconds"), "clock.max_pair_skew_seconds")
    asset(clock.get("synchronization_evidence"), "clock.synchronization_evidence")
    calibration = object_field(plan.get("calibration"), "calibration")
    text_field(calibration.get("method"), "calibration.method")
    for field in ("model_evidence", "parameter_covariance_evidence"):
        asset(calibration.get(field), f"calibration.{field}")
    session_groups = []
    for field, ids in (("calibration.session_ids", calibration.get("session_ids")),
                       ("development_session_ids", plan.get("development_session_ids"))):
        if not isinstance(ids, list) or not ids:
            issue("missing", field, "List separate whole calibration/development session IDs.")
            ids = []
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            issue("error", field, "Session IDs must be nonempty strings.")
            ids = []
        if len(set(ids)) != len(ids) or (recording_id is not None and recording_id in ids):
            issue("error", field, "Session IDs must be unique and disjoint from this held-out recording.")
        session_groups.append(set(ids))
    if session_groups[0] & session_groups[1]:
        issue("error", "session_ids", "Calibration and development sessions must be distinct in this pilot protocol.")
    asset(plan.get("locked_configuration_evidence"), "locked_configuration_evidence")
    uncertainty = object_field(plan.get("uncertainty"), "uncertainty")
    text_field(uncertainty.get("reference_dependence"), "uncertainty.reference_dependence")
    for field in ("joint_covariance_evidence", "reference_dependence_evidence"):
        asset(uncertainty.get(field), f"uncertainty.{field}")
    acceptance = object_field(plan.get("acceptance"), "acceptance")
    nonnegative(acceptance.get("maximum_reference_discrepancy_m"), "acceptance.maximum_reference_discrepancy_m")
    asset(acceptance.get("criterion_evidence"), "acceptance.criterion_evidence")
    videos = plan.get("raw_videos")
    if not isinstance(videos, list) or not videos:
        issue("missing", "raw_videos", "List retained original recordings as objects with id, path and sha256.")
        videos = []
    video_ids = set()
    video_keys, original_paths, original_digests = {}, set(), set()
    for index, video in enumerate(videos):
        video = object_field(video, f"raw_videos[{index}]")
        identifier = text_field(video.get("id"), f"raw_videos[{index}].id")
        if identifier in video_ids:
            issue("error", "raw_videos", "Original recording IDs must be unique.")
        if identifier is not None:
            video_ids.add(identifier)
        checked_asset = asset(video, f"raw_videos[{index}]")
        if checked_asset is not None:
            original_path, digest = checked_asset
            if original_path in original_paths or digest in original_digests:
                issue("error", "raw_videos", "Register identical original file bytes once in this pilot; aliases cannot supply additional captured frames.")
            original_paths.add(original_path)
            original_digests.add(digest)
            if identifier is not None:
                video_keys[identifier] = digest

    clocks = {name: {} for name in TABLES}
    for name, (relative, columns) in TABLES.items():
        try:
            path = relative_path(relative)
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != list(columns):
                    raise ValueError(f"CSV header must be exactly {','.join(columns)}")
                rows = list(reader)
        except (OSError, ValueError, csv.Error) as error:
            issue("missing" if isinstance(error, FileNotFoundError) else "error", relative, str(error))
            continue
        tables[name] = rows
        if not rows:
            issue("missing", relative, "No recorded observations are present.")
        previous_time, previous_counter = None, None
        source_positions, decoder_counters = {}, {}
        for number, row in enumerate(rows, 2):
            field = f"{relative}:{number}"
            if None in row or any(value is None for value in row.values()):
                issue("error", field, "CSV row does not match its header.")
                continue
            identifier = text_field(row["sample_id"], field + ".sample_id")
            if identifier is None:
                continue
            if identifier in clocks[name]:
                issue("error", field, "Duplicate sample ID; observations are not silently combined.")
                continue
            clocks[name][identifier] = row["capture_utc"] or None
            if name == "camera":
                if row["present"] not in ("true", "false"):
                    issue("error", field, "present must be true or false.")
                present = row["present"] == "true"
                source_key = video_keys.get(row["source_video_id"], row["source_video_id"])
                frame_counter = (None if counter_kind == "unavailable" and row["frame_id"] == ""
                                 else counter(row["frame_id"], field + ".frame_id"))
                if frame_counter is not None:
                    previous = decoder_counters.get(source_key) if counter_kind == "decoder_index" else previous_counter
                    if previous is not None and frame_counter <= previous:
                        issue("error", field, "Frame counters must increase in original acquisition/source order.")
                    previous_counter = frame_counter
                    decoder_counters[source_key] = frame_counter
                if present:
                    asset({"path": row["frame_path"], "sha256": row["frame_sha256"]}, field + ".frame")
                    source_index = counter(row["source_frame_index"], field + ".source_frame_index")
                    if row["source_video_id"] not in video_ids:
                        issue("error", field, "A frame must identify its retained original recording and source index.")
                    elif source_index is not None:
                        previous_source_index = source_positions.get(source_key)
                        if previous_source_index is not None and source_index <= previous_source_index:
                            issue("error", field, "Source frame indices must increase within each original recording; repeated extraction is not a new captured frame.")
                        source_positions[source_key] = source_index
                elif row["frame_path"] or row["frame_sha256"]:
                    issue("error", field, "A missing frame cannot have a substituted payload.")
                if not present:
                    clocks[name][identifier] = None
                    if row["capture_utc"]:
                        issue("review", field, "Missing payload retains a declared capture timestamp; it is excluded from clock triples.")
            else:
                present = bool(row["level_m"])
                if present:
                    try:
                        if not math.isfinite(float(row["level_m"])):
                            raise ValueError()
                    except ValueError:
                        issue("error", field, "Level must be a finite number in metres; blank means missing.")
                else:
                    clocks[name][identifier] = None
                if present or row["standard_uncertainty_m"]:
                    nonnegative(row["standard_uncertainty_m"], field + ".standard_uncertainty_m")
            start_key, end_key = (("exposure_start_utc", "exposure_end_utc") if name == "camera"
                                  else ("support_start_utc", "support_end_utc"))
            try:
                capture, start, end = (parse_capture_utc(row[key]) if present or row[key] else None
                                       for key in ("capture_utc", start_key, end_key))
                if ((start is not None and end is not None and start > end)
                        or (capture is not None and start is not None and capture < start)
                        or (capture is not None and end is not None and capture > end)):
                    raise ValueError("Capture timestamp must lie within ordered declared temporal support.")
                if capture is not None:
                    if previous_time is not None and capture <= previous_time:
                        raise ValueError("Capture times must increase in original row order.")
                    previous_time = capture
            except ValueError as error:
                issue("error", field + ".time", str(error))
    if skew_limit is not None:
        try:
            synchronization = compare_capture_clocks(**clocks, max_skew_seconds=skew_limit)
            if synchronization["common_samples"] == 0:
                issue("missing", "synchronization", "No complete camera/gauge/reference capture triple is available.")
            if synchronization["exceeded_sample_ids"]:
                issue("error", "synchronization", "Some declared timestamp differences exceed the stated skew limit; no alignment was performed.")
        except ValueError as error:
            issue("error", "synchronization", str(error))
    return result()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="Create blank planning files without overwriting evidence")
    initialize.add_argument("directory", type=Path)
    check = commands.add_parser("check", help="Inspect metadata, original files and capture-clock declarations")
    check.add_argument("directory", type=Path)
    check.add_argument("--out-report", type=Path)
    check.add_argument("--json", action="store_true", help="Print the complete machine-readable report")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(initialize_recording(args.directory).resolve())
            return 0
        report = check_recording(args.directory)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out_report is not None:
            if args.out_report.resolve().is_relative_to(args.directory.resolve()):
                raise ValueError("Write preflight reports outside the recording folder to preserve source evidence.")
            args.out_report.parent.mkdir(parents=True, exist_ok=True)
            args.out_report.write_text(text, encoding="utf-8")
        if args.json:
            print(text, end="")
        else:
            print(f"Recording preflight: {report['status']}")
            print("No field validation or quantitative evaluation has been performed.")
            print("Rows: " + ", ".join(f"{name}={count}" for name, count in report["rows"].items()))
            for item in report["issues"][:10]:
                print(f"  {item['kind']}: {item['field']}: {item['message']}")
            if len(report["issues"]) > 10:
                print(f"  {len(report['issues']) - 10} more items; use --json or --out-report for the full list.")
        return 0 if report["status"] == "ready_for_review" else 2
    except (OSError, ValueError) as error:
        parser.exit(2, f"recording: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
