"""Fetch a window of NOAA water levels through DAF's own adapter, recording every response.

One of only two tools in this repository that make a network request (the other is
`tools/fetch_usgs_reservoir.py`); neither is invoked by a test, and neither is needed to
reproduce anything. This one writes `data/daf/raw/<session>/`, and from there
`tools/export_daf_fixtures.py` replays the recorded bytes through the same DAF binding
offline, byte for byte, forever. Fetching again is only ever needed to acquire a
*different* window.

What runs. DAF's own incremental acquisition, unmodified: the per-measurement NOAA binding
(`daf.orchestration.bindings.noaa_water_level_measurement_binding`, i.e.
`NoaaWaterLevelSourceAdapter` paired with `NoaaWaterLevelMeasurementExtractor`) driven by
`daf.scheduling.runner.execute_plan` against a `CheckpointStore`, called until the plan
stops reporting new data. The adapter fetches a bounded window per call (`window_days` = 3)
and re-requests a trailing safety window (`revision_lookback_days` = 2) each time, because
a NOAA window's bytes can change when QC flips readings from preliminary to verified. So
consecutive requests OVERLAP by design and most readings are acquired several times. That
is DAF's behaviour, not this tool's: the overlap is recorded as it happened, and the
bridge's deduplication -- identical content, every id kept, the reading enters once -- is
what resolves it downstream.

The adapter's own `fetch_bytes` injection point tees each response to disk. The bytes
written are exactly what the socket delivered, before any parsing; nothing is normalised,
reordered or re-encoded. Each response is stored under the sha256 of its own bytes, with
`index.json` mapping the request URL to it, so a replay can refuse an unrecorded URL
rather than silently fetching.

    uv run --python 3.13 python tools/fetch_noaa_month.py \
        --station 8454000 --begin 20240101 --end 20240131 --datum MLLW --units metric \
        --requested-at 2026-09-12T00:00:00Z --session noaa_8454000_202401_mllw

`--requested-at` is DAF's caller-supplied retrieval stamp (`run_scout` copies it into
`extracted_at`); it is never wall-clock, so re-running with the same value reproduces the
same evidence ids. `DAF_ROOT` must point at a DAF checkout at the pinned commit with its
vendored substrate initialised, exactly as the exporter requires.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from _daf_session import (
    DEFAULT_RAW, MAX_CALLS, Recorder, base_index, check_requested_at, daf_root, import_daf,
    prepare_session, run_plan_to_exhaustion, write_index,
)

SOURCE_ID = "noaa-water-level-measurements"   # the operator's DAF source id; not part of any evidence id
PRODUCT = "water_level"
USER_AGENT = "fluid-state-reconstruction-testbed/0.1 (research; one-off archival fetch)"


def _windows_expected(begin: str, end: str, window_days: int, lookback_days: int) -> list[tuple[str, str]]:
    """The windows DAF's adapter will request, derived from its own rule, so the tool can say
    up front how many live requests a run will make.

    The rule advances by (window_days - lookback_days + 1) days per call and clamps the last
    window to the scope end, so once a window ends at the scope end the NEXT call requests
    the same window again. DAF terminates there on its own: the same locator re-fetched to
    the same bytes is one artifact version already in the pool, so the plan reports DUPLICATE
    rather than ACQUIRED. That repeat is listed too -- it is a real request this tool makes,
    though it is served from what was already recorded rather than fetched twice.
    """
    d0, d1 = (datetime.strptime(s, "%Y%m%d").date() for s in (begin, end))
    if d1 < d0:
        raise SystemExit(f"--end {end} is before --begin {begin}")
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    prev_end: date | None = None
    while True:
        start = d0 if prev_end is None else prev_end - timedelta(days=lookback_days - 1)
        if start > d1:
            return out
        w_end = min(start + timedelta(days=window_days - 1), d1)
        w = (start.strftime("%Y%m%d"), w_end.strftime("%Y%m%d"))
        out.append(w)
        if w in seen:                       # the repeat DAF answers with DUPLICATE: the loop ends here
            return out
        seen.add(w)
        prev_end = w_end
        if len(out) > MAX_CALLS:
            raise SystemExit(f"more than {MAX_CALLS} windows for {begin}..{end}; refusing")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", required=True)
    ap.add_argument("--begin", required=True, help="YYYYMMDD, inclusive")
    ap.add_argument("--end", required=True, help="YYYYMMDD, inclusive")
    ap.add_argument("--datum", default="MLLW")
    ap.add_argument("--units", default="metric")
    ap.add_argument("--requested-at", required=True, help="DAF's retrieval stamp, ISO-8601 UTC; never wall-clock")
    ap.add_argument("--session", required=True, help="directory name under data/daf/raw/")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--dry-run", action="store_true", help="print the windows that would be requested; fetch nothing")
    args = ap.parse_args(argv)
    requested_at = check_requested_at(args.requested_at)

    root = daf_root("daf/adapters/noaa_water_level.py")
    d = import_daf(root)
    from daf.adapters.noaa_water_level import (          # noqa: PLC0415 -- needs DAF_ROOT on sys.path first
        _DEFAULT_REVISION_LOOKBACK_DAYS, _DEFAULT_WINDOW_DAYS, DATAGETTER_BASE,
    )
    from daf.orchestration.bindings import noaa_water_level_measurement_binding  # noqa: PLC0415

    expected = _windows_expected(args.begin, args.end, _DEFAULT_WINDOW_DAYS, _DEFAULT_REVISION_LOOKBACK_DAYS)
    print(f"DAF's adapter will request {len(expected)} window(s) of {_DEFAULT_WINDOW_DAYS} day(s) "
          f"(trailing lookback {_DEFAULT_REVISION_LOOKBACK_DAYS}), covering {args.begin}..{args.end}:")
    print(f"  first {expected[0][0]}..{expected[0][1]}, last {expected[-1][0]}..{expected[-1][1]}")
    if args.dry_run:
        return 0

    out_dir = prepare_session(args.raw_dir, args.session)

    def check(url: str) -> dict:
        """Every request the adapter makes is checked against what this invocation asked
        for; a recorded session is evidence of what was intended only if it was."""
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        want = {"product": PRODUCT, "station": args.station, "datum": args.datum, "units": args.units,
                "time_zone": "gmt", "format": "json"}
        bad = {k: (q.get(k), v) for k, v in want.items() if q.get(k) != v}
        if bad or not url.startswith(DATAGETTER_BASE + "?"):
            raise RuntimeError(f"the adapter requested {url!r}; mismatched {bad}")
        return {"begin_date": q["begin_date"], "end_date": q["end_date"]}

    recorder = Recorder(out_dir=out_dir, user_agent=USER_AGENT, check=check,
                        describe=lambda b: {"n_items": len(json.loads(b).get("data", []))})
    binding = noaa_water_level_measurement_binding(datum=args.datum, units=args.units, fetch_bytes=recorder)
    params = {"station": args.station, "product": PRODUCT, "start_date": args.begin, "end_date": args.end}
    outcomes, n_obs = run_plan_to_exhaustion(
        d, binding=binding, source_id=SOURCE_ID, source_name="NOAA CO-OPS Tides & Currents",
        required_parameters=("station", "product", "start_date", "end_date"),
        plan_id="fsre-noaa-month", parameters=params, requested_at=requested_at)

    index = {
        **base_index(root, session=args.session, tool="tools/fetch_noaa_month.py", binding=binding,
                     requested_at=requested_at, parameters=params),
        "binding_factory": "daf.orchestration.bindings.noaa_water_level_measurement_binding",
        "window_days": _DEFAULT_WINDOW_DAYS,
        "revision_lookback_days": _DEFAULT_REVISION_LOOKBACK_DAYS,
        "datum": args.datum, "units": args.units, "time_zone": "gmt",
        "n_requests": len(recorder.responses),
        "n_observations_admitted": n_obs,
        "outcomes": outcomes,
        "windows_expected": [{"begin_date": b, "end_date": e} for b, e in expected],
        "responses": recorder.responses,
        "note": "Each response is the raw HTTPS body, byte for byte, stored under the sha256 of its own "
                "bytes. Consecutive windows overlap because DAF's adapter re-requests a trailing safety "
                "window; the overlap is recorded as it happened and resolved downstream by the bridge's "
                "deduplication, which compares whole contents.",
    }
    write_index(out_dir, index)
    print(f"\nwrote {out_dir}: {len(recorder.responses)} response(s), {n_obs} observations admitted, "
          f"outcomes {outcomes}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
