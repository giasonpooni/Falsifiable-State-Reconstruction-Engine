"""Fetch a reservoir's daily-value series through DAF's own adapter, recording every response.

The second of the two tools here that reach the network (see `tools/fetch_noaa_month.py`
for the discipline they share: DAF's own acquisition drives it, responses are teed to
`data/daf/raw/<session>/` under the sha256 of their own bytes before any parsing, and
`tools/export_daf_fixtures.py` replays them offline forever afterwards).

What it acquires. One `--series SITE:PARAM:STAT` argument per daily series, each its own
DAF plan against the same recorded session, so a reservoir's storage, outflow and inflows
arrive as several independent series rather than one joined table. Joining them -- deciding
that a storage value and a flow value describe the same day -- is a declaration the
consumer makes, with its own citation, not something this tool bakes into the evidence.
That matters here: the storage series is a daily MEAN and the flows are daily MEANs too, so
they are half a day out of step with an end-of-day storage, and which alignment is right is
an argument, not a fact.

    uv run --python 3.13 python tools/fetch_usgs_reservoir.py \
        --series USGS-09147022:00054:00003 --series USGS-09147025:00060:00003 \
        --series USGS-09146200:00060:00003 --series USGS-09147000:00060:00003 \
        --begin 2022-10-01 --end 2025-09-30 \
        --requested-at 2026-09-12T00:00:00Z --session usgs_ridgway_wy2023_2025

`--requested-at` is DAF's caller-supplied retrieval stamp, never wall-clock, so a re-run
reproduces the same evidence ids. `DAF_ROOT` must point at a DAF checkout carrying the
USGS daily-values adapter (`daf/adapters/usgs_daily_values.py`), at the pinned commit with
its vendored substrate initialised. That adapter is not in DAF's published history: it is
carried in this repository as a patch series under `patches/daf/`, together with the
fixtures it exports.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from _daf_session import (
    DEFAULT_RAW, MAX_CALLS, Recorder, base_index, check_requested_at, daf_root, import_daf,
    prepare_session, run_plan_to_exhaustion, write_index,
)

SOURCE_ID = "usgs-daily-values"
USER_AGENT = "Data-Acquisition-Fabric research-adapter contact@example.com"
_SERIES_RE = re.compile(r"\A(?P<site>[A-Za-z0-9_.:-]+):(?P<param>\d{5}):(?P<stat>\d{5})\Z")


def _parse_series(text: str) -> tuple[str, str, str]:
    m = _SERIES_RE.match(text)
    if m is None:
        raise SystemExit(f"--series must be SITE:PARAMETER_CODE:STATISTIC_ID (e.g. "
                         f"USGS-09147022:00054:00003); got {text!r}")
    return m.group("site"), m.group("param"), m.group("stat")


def _windows_expected(begin: str, end: str, window_days: int, lookback_days: int) -> list[tuple[str, str]]:
    """The windows DAF's daily-values adapter will request. Same rule as the NOAA adapter's,
    and the same terminating repeat: once a window ends at the scope end, the next call asks
    for it again, DAF answers DUPLICATE and the catch-up stops."""
    d0, d1 = (date.fromisoformat(s) for s in (begin, end))
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
        w = (start.isoformat(), w_end.isoformat())
        out.append(w)
        if w in seen:
            return out
        seen.add(w)
        prev_end = w_end
        if len(out) > MAX_CALLS:
            raise SystemExit(f"more than {MAX_CALLS} windows for {begin}..{end}; refusing")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", action="append", required=True, metavar="SITE:PARAM:STAT",
                    help="a daily series; repeat for each one (order is the order recorded)")
    ap.add_argument("--begin", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--requested-at", required=True, help="DAF's retrieval stamp, ISO-8601 UTC; never wall-clock")
    ap.add_argument("--session", required=True, help="directory name under data/daf/raw/")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--dry-run", action="store_true", help="print what would be requested; fetch nothing")
    args = ap.parse_args(argv)
    requested_at = check_requested_at(args.requested_at)
    for s in (args.begin, args.end):
        try:
            date.fromisoformat(s)
        except ValueError:
            raise SystemExit(f"dates must be YYYY-MM-DD; got {s!r}") from None
    series = [_parse_series(s) for s in args.series]
    if len({s for s in series}) != len(series):
        raise SystemExit(f"--series lists one series twice: {args.series}")

    root = daf_root("daf/adapters/usgs_daily_values.py", "daf/extractors/usgs_daily_values.py")
    d = import_daf(root)
    from daf.adapters.usgs_daily_values import (        # noqa: PLC0415 -- needs DAF_ROOT on sys.path first
        DAILY_ITEMS_BASE, PAGE_LIMIT, _DEFAULT_REVISION_LOOKBACK_DAYS, _DEFAULT_WINDOW_DAYS,
    )
    from daf.orchestration.bindings import usgs_daily_values_binding  # noqa: PLC0415

    expected = _windows_expected(args.begin, args.end, _DEFAULT_WINDOW_DAYS, _DEFAULT_REVISION_LOOKBACK_DAYS)
    print(f"{len(series)} series x {len(expected)} window(s) of up to {_DEFAULT_WINDOW_DAYS} day(s) "
          f"(trailing lookback {_DEFAULT_REVISION_LOOKBACK_DAYS}, page limit {PAGE_LIMIT}), "
          f"covering {args.begin}..{args.end}:")
    for site, param, stat in series:
        print(f"  {site} parameter {param} statistic {stat}")
    print(f"  windows: first {expected[0][0]}..{expected[0][1]}, last {expected[-1][0]}..{expected[-1][1]}")
    if args.dry_run:
        return 0

    out_dir = prepare_session(args.raw_dir, args.session)
    per_series = []
    all_responses: list[dict] = []

    for site, param, stat in series:
        def check(url: str, site=site, param=param, stat=stat) -> dict:
            q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
            want = {"monitoring_location_id": site, "parameter_code": param, "statistic_id": stat,
                    "limit": str(PAGE_LIMIT), "f": "json"}
            bad = {k: (q.get(k), v) for k, v in want.items() if q.get(k) != v}
            if bad or not url.startswith(DAILY_ITEMS_BASE + "?"):
                raise RuntimeError(f"the adapter requested {url!r}; mismatched {bad}")
            begin, _, end = q["datetime"].partition("/")
            return {"monitoring_location_id": site, "parameter_code": param, "statistic_id": stat,
                    "begin_date": begin, "end_date": end}

        recorder = Recorder(out_dir=out_dir, user_agent=USER_AGENT, check=check,
                            describe=lambda b: {"n_items": len(json.loads(b).get("features", []))})
        binding = usgs_daily_values_binding(fetch_bytes=recorder)
        params = {"monitoring_location_id": site, "parameter_code": param, "statistic_id": stat,
                  "start_date": args.begin, "end_date": args.end}
        print(f"\n{site} {param}/{stat}:")
        outcomes, n_obs = run_plan_to_exhaustion(
            d, binding=binding, source_id=SOURCE_ID, source_name="USGS Water Data OGC API -- daily values",
            required_parameters=("monitoring_location_id", "parameter_code", "statistic_id",
                                 "start_date", "end_date"),
            plan_id=f"fsre-usgs-{site}-{param}-{stat}", parameters=params, requested_at=requested_at)
        per_series.append({"monitoring_location_id": site, "parameter_code": param, "statistic_id": stat,
                           "plan_parameters": params, "outcomes": outcomes,
                           "n_requests": len(recorder.responses), "n_observations_admitted": n_obs,
                           "n_items_total": sum(r.get("n_items", 0) for r in recorder.responses)})
        all_responses.extend(recorder.responses)

    index = {
        **base_index(root, session=args.session, tool="tools/fetch_usgs_reservoir.py", binding=binding,
                     requested_at=requested_at, parameters={"begin": args.begin, "end": args.end}),
        "binding_factory": "daf.orchestration.bindings.usgs_daily_values_binding",
        "window_days": _DEFAULT_WINDOW_DAYS,
        "revision_lookback_days": _DEFAULT_REVISION_LOOKBACK_DAYS,
        "page_limit": PAGE_LIMIT,
        "scope": {"begin": args.begin, "end": args.end},
        "series": per_series,
        "n_requests": len(all_responses),
        "windows_expected": [{"begin_date": b, "end_date": e} for b, e in expected],
        "responses": all_responses,
        "note": "One DAF plan per series: storage, outflow and inflows are acquired as independent "
                "series and never joined here. Deciding that a storage value and a flow value describe "
                "the same day is a consumer's declaration -- the storage series is a daily MEAN, so it "
                "is half a day out of step with an end-of-day reading, and which alignment is right is "
                "an argument rather than a fact. Each response is the raw HTTPS body, byte for byte.",
    }
    write_index(out_dir, index)
    print(f"\nwrote {out_dir}: {len(all_responses)} response(s) over {len(series)} series")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
