"""Interval-aware, measurement-space replay of the committed Ridgway record.

This is a conditional consistency calculation, not a sensor-fault validation. Daily
mean storage needs an explicit within-day model; this replay declares constant net flow
within each source day. Its uncertainty scales are the existing consumer-declared sweep.
No fitted state, repeated pseudo-measurement, or independently redrawn initial reference
enters the calculation. Every residual and its joint covariance use the same raw record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..bridge.daf import BridgedSeries
from ..diagnostics import diagnose
from ..measurement import BalanceRecord, balance_residuals
from . import real_water_balance as wb
from .provenance import REPO_ROOT, header_line, provenance

CFS_TO_M3_S = 0.3048 ** 3
ACRE_FT_TO_M3 = 43560.0 * CFS_TO_M3_S
WINDOW_DAYS = 32
# The declared window length is a consumer choice and it moves the headline more than the
# declared storage sigma does, so it is swept rather than fixed. 3 days is the shortest
# window with two residuals; 64 is twice the declared length. Longer windows leave fewer,
# coarser cells (at 64 days this record gives 18), which is a limit of the record, not a
# result about it.
WINDOW_DAYS_SENSITIVITY = (3, 8, WINDOW_DAYS, 64)


def record_from_bridge(bs: BridgedSeries, start: int, stop: int) -> BalanceRecord:
    """Convert a complete contiguous source window to SI, preserving component evidence.

    Source R can include within-sample correlations. Cross-day measurement errors are
    declared independent for this replay; shared-reference dependence is induced by the
    residual operator, not discarded. A caller with cross-day calibration covariance
    should supply that joint covariance directly to BalanceRecord instead.
    """
    if (any(isinstance(i, bool) or not isinstance(i, int) for i in (start, stop)) or
            not (0 <= start < stop <= len(bs.observations)) or stop - start < 2):
        raise ValueError("a replay window needs at least two observations within the record")
    if tuple(bs.source_ids) != wb.SOURCE_IDS:
        raise ValueError("the ordered storage/outflow/inflow source identities must match")
    if bs.provenance.get("selectors") != [selector.describe() for selector in wb.SELECTORS]:
        raise ValueError("the replay requires the declared USGS daily-mean quantities and original units")
    declared_time = bs.provenance.get("time", {})
    required_time = {"time_semantics": "calendar_day", "time_zone": "UTC",
                     "day_anchor": "midpoint", "day_anchor_offset_s": 43200.0}
    if (any(declared_time.get(key) != value for key, value in required_time.items()) or
            not declared_time.get("day_zone_citation") or bs.provenance.get("cadence_s") != 86400.0):
        raise ValueError("the replay requires cited UTC daily intervals anchored at their midpoint")
    obs = bs.observations[start:stop]
    times = np.asarray(bs.inputs.t[start:stop], dtype=float)
    if (times.shape != (len(obs),) or not np.all(np.isfinite(times)) or
            not np.allclose(np.diff(times), 86400.0, rtol=0.0, atol=1e-9)):
        raise ValueError("the replay requires contiguous declared 86400-second days")
    n = len(obs)
    values = np.empty((n, 4))
    covariance = np.zeros((4 * n, 4 * n))
    scales = np.array([ACRE_FT_TO_M3, CFS_TO_M3_S, CFS_TO_M3_S, CFS_TO_M3_S])
    for k, o in enumerate(obs):
        if tuple(o.source_ids) != wb.SOURCE_IDS or o.t != times[k]:
            raise ValueError("observation identities or sample clock disagree with the declared record")
        mask, reading, raw_covariance = np.asarray(o.mask), np.asarray(o.y), np.asarray(o.R)
        if (mask.shape != (4,) or mask.dtype.kind != "b" or reading.shape != (4,) or
                raw_covariance.shape != (4, 4) or np.iscomplexobj(reading) or
                np.iscomplexobj(raw_covariance)):
            raise ValueError("each observation needs four real readings, a 4x4 covariance and four boolean masks")
        if not mask.all():
            raise ValueError("incomplete measurement window: no interpolation or implicit zero")
        values[k] = reading * scales
        indices = [k, n + 3 * k, n + 3 * k + 1, n + 3 * k + 2]
        covariance[np.ix_(indices, indices)] = raw_covariance * np.outer(scales, scales)
    evidence = tuple(bs.component_evidence[k][0] for k in range(start, stop)) + tuple(
        bs.component_evidence[k][j] for k in range(start, stop) for j in (1, 2, 3)
    )
    return BalanceRecord(
        edges=np.r_[times - 43200.0, times[-1] + 43200.0],
        storage=values[:, 0], flows=values[:, 1:], flow_signs=np.array([-1.0, 1.0, 1.0]),
        covariance=covariance, storage_support="mean", flow_support="mean",
        within_interval_model="constant_net_flow", volume_unit="m3", flow_unit="m3/s",
        evidence_ids=evidence,
    )


def _scan(bs: BridgedSeries, window_days: int) -> tuple[list[dict], list[dict]]:
    """Every nonoverlapping window of the record at one declared length.

    The window length is a consumer choice, not a property of the record, and it is
    not neutral: a longer window accumulates more of a persistent misfit while its
    chi-square threshold grows more slowly, so the same measurements reject more often
    at a longer window. `window_sensitivity` in the report measures that.
    """
    windows, skipped = [], []
    for start in range(0, len(bs.observations), window_days):
        stop = min(start + window_days, len(bs.observations))
        if stop - start < 2:
            skipped.append({"start": start, "stop": stop, "reason": "fewer than two days"})
            continue
        if not all(np.asarray(o.mask, dtype=bool).all() for o in bs.observations[start:stop]):
            skipped.append({"start": start, "stop": stop, "reason": "missing measurement"})
            continue
        record = record_from_bridge(bs, start, stop)
        adjacent = balance_residuals(record)
        cumulative = balance_residuals(record, cumulative=True)
        local = diagnose(adjacent.residual, adjacent.covariance, {})
        joint = diagnose(cumulative.residual, cumulative.covariance, {})
        # Invertible cumulative summation cannot add independent evidence.
        if not np.isclose(local.null_statistic, joint.null_statistic, rtol=1e-8, atol=1e-8):
            raise AssertionError("adjacent/cumulative joint statistics disagree")
        ids = sorted({identifier for group in cumulative.evidence_ids for identifier in group})
        windows.append({
            "start": start, "stop": stop, "n_days": stop - start,
            "n_residuals": len(adjacent.residual), "status": joint.status,
            "joint_statistic": joint.null_statistic, "dof": joint.null_dof,
            "threshold": joint.null_threshold,
            "adjacent_joint_statistic": local.null_statistic,
            "final_cumulative_residual_m3": float(cumulative.residual[-1]),
            "final_cumulative_std_m3": float(np.sqrt(cumulative.covariance[-1, -1])),
            "shared_initial_reading_variance_m6": float(record.covariance[0, 0]),
            "first_two_cumulative_covariance_m6": (
                float(cumulative.covariance[0, 1]) if len(cumulative.residual) > 1 else None
            ),
            "evidence_ids": ids,
        })
    return windows, skipped


def _aggregate(bs: BridgedSeries, windows: list[dict], skipped: list[dict], window_days: int) -> dict:
    """The counts a reader compares across cells, and no per-window detail.

    statistic_over_threshold is reported because the rejected COUNT is a step function
    of a continuous quantity: a cell can have no rejections while its typical window
    sits just under its threshold, which a count alone hides.
    """
    ratios = sorted(w["joint_statistic"] / w["threshold"] for w in windows
                    if w["threshold"])
    tested = sum(w["n_residuals"] for w in windows)
    return {
        "window_days": window_days,
        "n_windows": len(windows),
        "n_skipped_windows": len(skipped),
        "rejected_windows": sum(w["status"] != "consistent" for w in windows),
        "rejected_fraction": (sum(w["status"] != "consistent" for w in windows) / len(windows)
                              if windows else None),
        "dof_per_window": windows[0]["dof"] if windows else None,
        "tested_adjacent_pairs": tested,
        "untested_adjacent_pairs": len(bs.observations) - 1 - tested,
        "statistic_over_threshold": {
            "median": ratios[len(ratios) // 2] if ratios else None,
            "max": ratios[-1] if ratios else None,
        },
    }


def build_report(*, window_days: int = WINDOW_DAYS,
                 window_sensitivity: tuple[int, ...] = WINDOW_DAYS_SENSITIVITY) -> dict:
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 3:
        raise ValueError("window_days must be an integer >= 3")
    lengths = tuple(window_sensitivity)
    if (any(isinstance(w, bool) or not isinstance(w, int) or w < 3 for w in lengths)
            or len(set(lengths)) != len(lengths)):
        raise ValueError("window_sensitivity must be distinct integers >= 3")
    if window_days not in lengths:
        raise ValueError("the declared window length must be one point of its own sensitivity sweep")
    sweeps, sensitivity = [], []
    for sigma in wb.STORAGE_SIGMA_SWEEP:
        bs = wb.load(sigma)
        windows, skipped = _scan(bs, window_days)
        for length in sorted(lengths):
            scanned = (windows, skipped) if length == window_days else _scan(bs, length)
            cell = _aggregate(bs, *scanned, length)
            cell["storage_sigma_acre_ft"] = sigma
            cell["is_declared_window"] = length == window_days
            sensitivity.append(cell)
        sweeps.append({
            "storage_sigma_acre_ft": sigma, "n_days": len(bs.observations),
            "bridge_provenance": bs.provenance,
            "windows": windows, "skipped_windows": skipped,
            "tested_adjacent_pairs": sum(w["n_residuals"] for w in windows),
            "untested_adjacent_pairs": len(bs.observations) - 1 - sum(w["n_residuals"] for w in windows),
            "rejected_windows": sum(w["status"] != "consistent" for w in windows),
        })
    out = {
        "schema_version": "fsre-real-fluid-baseline-v3",
        "scope": "Real-record conditional consistency; no known fault labels or ground truth.",
        "declared": {
            "window_days": window_days, "window_policy": "nonoverlapping, anchored at first source day",
            "window_days_sensitivity": list(sorted(lengths)),
            "window_days_status": "consumer choice; the declared length is one point of the swept axis "
                                  "in this report and is not privileged by the record",
            "alpha": 0.01, "storage_support": "interval mean", "flow_support": "interval mean",
            "within_interval_model": "constant_net_flow",
            "within_interval_model_status": "consumer assumption; not established by daily means",
            "day_zone_citation": wb.DAY_ZONE_CITATION,
            "flow_sigma_citation": wb.FLOW_SIGMA_CITATION,
            "storage_sigma_citation": wb.STORAGE_SIGMA_CITATION,
            "cross_day_measurement_covariance": "zero; not empirically calibrated",
            "shared_reference": "the same first storage reading in every cumulative row; joint covariance retained",
            "unit_conversion": {"acre_ft_to_m3": ACRE_FT_TO_M3, "cfs_to_m3_s": CFS_TO_M3_S},
        },
        "provenance": {"generation": provenance()},
        "sweeps": sweeps,
        "window_sensitivity": sensitivity,
        "limitations": [
            "A rejected window is not a verified sensor fault or a measured false alarm.",
            "Unmeasured water, varying within-day flow, calibration error and model error remain possible causes.",
            "The nominal per-window threshold is conditional on Gaussian errors and the declared covariance.",
            "No network-wide or repeated-window alarm probability is claimed.",
            "Cross-window adjacent pairs and incomplete windows are explicitly counted, not silently filled.",
            "The rejected-window rate is a joint property of the record and the declared window length; both axes are swept and neither is a measurement of the gauges alone.",
            "Legacy wb_open/wb_aug/wb_closed filters retain their documented approximation; this path works directly on measurements.",
        ],
    }
    out["claims"] = claims(out)
    return out


def claims(report: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition.

    If one fails the report is not written: the text would no longer be true of the numbers.
    The sentences that need it are the ones about the WINDOW AXIS -- that a longer window
    rejects more often on the same measurements, and that the axis moves the rate at least as
    far as the declared sigma does. Both are properties of this record.
    """
    cells = {(c["storage_sigma_acre_ft"], c["window_days"]): c for c in report["window_sensitivity"]}
    sigmas = sorted({sigma for sigma, _ in cells})
    lengths = sorted({length for _, length in cells})
    medians_rise = all(
        [cells[(sigma, length)]["statistic_over_threshold"]["median"] for length in lengths]
        == sorted(cells[(sigma, length)]["statistic_over_threshold"]["median"] for length in lengths)
        for sigma in sigmas)
    window_spread = max(cells[(sigmas[0], length)]["rejected_fraction"] for length in lengths) - \
        min(cells[(sigmas[0], length)]["rejected_fraction"] for length in lengths)
    sigma_spread = max(cells[(sigma, report["declared"]["window_days"])]["rejected_fraction"]
                       for sigma in sigmas) - \
        min(cells[(sigma, report["declared"]["window_days"])]["rejected_fraction"] for sigma in sigmas)
    return {
        "the_declared_window_is_one_point_of_the_swept_axis":
            report["declared"]["window_days"] in lengths,
        "a_longer_window_raises_the_median_statistic_over_its_threshold": medians_rise,
        "the_window_axis_moves_the_rate_at_least_as_far_as_the_sigma_axis":
            window_spread >= sigma_spread,
        "a_wider_declared_sigma_never_rejects_more": all(
            (lambda rates: rates == sorted(rates, reverse=True))(
                [cells[(sigma, length)]["rejected_fraction"] for sigma in sigmas])
            for length in lengths),
        "every_cell_reports_how_close_its_median_window_was": all(
            c["statistic_over_threshold"]["median"] is not None for c in cells.values()),
        "the_declared_cell_agrees_with_the_detailed_sweep": all(
            cells[(sweep["storage_sigma_acre_ft"], report["declared"]["window_days"])]["rejected_windows"]
            == sweep["rejected_windows"] for sweep in report["sweeps"]),
    }


def render(report: dict) -> str:
    failed = [name for name, held in report.get("claims", {}).items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    lines = ["# Real fluid-measurement baseline — Ridgway", "",
             header_line(report["provenance"]["generation"]), "", report["scope"], "",
             "Storage and flow are interpreted as matching UTC daily means under the inherited consumer",
             "time-zone assumption. Constant net flow within each day is also a consumer assumption.",
             "The calculation propagates the joint measurement covariance",
             "through the balance operator, including the shared initial reading in cumulative residuals.", "",
             f"The detailed sweep immediately below uses windows of at most "
             f"{report['declared']['window_days']} days; that length is itself swept in the table "
             "after it. No sensor cause is assigned.", "",
             "| Storage sigma [acre-ft] | Tested windows | Rejected windows | Tested adjacent pairs | Untested pairs |",
             "|---|---:|---:|---:|---:|"]
    for sweep in report["sweeps"]:
        lines.append(f"| {sweep['storage_sigma_acre_ft']:g} | {len(sweep['windows'])} | "
                     f"{sweep['rejected_windows']} | {sweep['tested_adjacent_pairs']} | {sweep['untested_adjacent_pairs']} |")
    dec = report["declared"]
    lines += ["", "## The window length decides the rate as much as the declared sigma does", "",
              "The window length is a consumer choice. A longer window accumulates more of a",
              "persistent misfit while its chi-square threshold grows more slowly, so the same",
              "measurements reject more often at a longer window. Both axes are therefore swept,",
              "and the median window's statistic as a fraction of its own threshold is reported",
              "beside each count: a cell with no rejections whose typical window sits just under",
              "threshold is not the same finding as one whose typical window sits far below it.", "",
              "| Storage sigma | Window days | Windows | Rejected | Rate | dof/window | median stat/threshold | max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in report["window_sensitivity"]:
        ratio = cell["statistic_over_threshold"]
        declared = " *(declared)*" if cell["is_declared_window"] else ""
        rate = "n/a" if cell["rejected_fraction"] is None else f"{cell['rejected_fraction'] * 100:.1f}%"
        lines.append(
            f"| {cell['storage_sigma_acre_ft']:g} | {cell['window_days']}{declared} | "
            f"{cell['n_windows']} | {cell['rejected_windows']} | {rate} | {cell['dof_per_window']} | "
            f"{'n/a' if ratio['median'] is None else format(ratio['median'], '.3f')} | "
            f"{'n/a' if ratio['max'] is None else format(ratio['max'], '.3f')} |")
    lines += ["",
              "Neither axis is a measurement of the gauges alone. Reading any one cell as the",
              "record's rejection rate reads a consumer choice as a property of the instruments.", "",
              "## Interpretation", "",
              "**Every uncertainty and interval here is consumer-declared and carries its citation,",
              "printed where the numbers are read and not only stored in the JSON.**", "",
              f"- Declared time interval: {dec['day_zone_citation']}",
              f"- Declared flow sigma: {dec['flow_sigma_citation']}",
              f"- Declared storage sigma: {dec['storage_sigma_citation']}",
              f"- Within-interval model (`{dec['within_interval_model']}`): "
              f"{dec['within_interval_model_status']}.",
              f"- Window length ({dec['window_days']} days, swept over "
              f"{dec['window_days_sensitivity']}): {dec['window_days_status']}.", "",
              "Uncertainty is the existing consumer-declared sensitivity sweep, not a field calibration.",
              "The JSON retains every window's joint statistic, threshold, final cumulative residual,",
              "conditional standard deviation, shared-reference covariance and supporting evidence IDs.",
              "Adjacent and cumulative residuals give the same joint statistic: cumulative summation",
              "does not create independent evidence.", ""]
    lines += [f"- {line}" for line in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = build_report()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "real_fluid_baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out_dir / "real_fluid_baseline.md").write_text(render(report), encoding="utf-8")
    if not quiet:
        print(render(report))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
