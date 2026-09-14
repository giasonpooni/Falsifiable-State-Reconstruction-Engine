"""A second reservoir, and what a second reservoir cost.

results/real_water_balance studies Ridgway. When its site was moved into a declaration, the
claim made was that a SECOND reservoir would cost a declaration rather than a module. This is
the test of that claim, and the honest answer is: a declaration, plus two one-time costs that
only a second site could have exposed.

    the declaration            declarations/taylor_park.toml. Same schema, nothing added.
    the FILTERS                estimators_balance assumed exactly two gauged inflows --
                               `N_SENSORS = 4` and three hand-indexed state layouts. That is a
                               property of Ridgway, not of a water balance, and no test could
                               have found it because every test had two inflows. Taylor Park
                               has three. The filters now take the count from BalanceConfig.
    the site view              experiments/balance_site.Site, so the study's functions read a
                               declaration instead of this module's globals.

Neither one-time cost is paid again by a third site. Both were invisible until a site with a
different shape was actually attempted, which is the whole reason for attempting one.

WHY THIS SITE. Taylor Park is deliberately not a clone. Three gauged inflows instead of two,
and two of the three are SEASONAL gauges that report on 642 of 1,096 days. Ridgway's four
series are complete -- every one of them, every day -- so the bridge's missing-reading path
had never been exercised by real evidence at all, only by synthetic tests. Here it carries 454
absent days on each of two columns.

WHAT IT FOUND, which is not what Ridgway found. Ridgway's balance closes: its three-year
cumulative residual is 0.41% of gauged inflow and its interval contains zero. Taylor Park's
does not close, and the shared claims() refused to print Ridgway's sentences about it -- which
is the guard doing its job on a site transfer rather than on a wording slip.

The report separates two explanations that would otherwise be confounded, because the record
makes it possible to:

    on the 641 days when EVERY gauge reports, the residual is 9.80% of gauged inflow
    on the 444 days when a seasonal creek is absent, it is larger

So the absent readings account for roughly half the apparent imbalance and no more. The other
half is present in fully observed data, and this study does not say what it is. It reports one
comparison and declines to draw the causal conclusion: 9.80% against an ungauged drainage
fraction of 8.90% is a close correspondence, and the same comparison at Ridgway is 0.41%
against 7.09%, which is not. A correspondence that holds at one site and fails at the other is
a coincidence or a mechanism, and two sites cannot tell which.

    python -m set_lcm.experiments.real_taylor_park  -> results/real_taylor_park.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..testbed.estimators_balance import CFS_DAY_TO_ACRE_FT
from . import real_water_balance as wb
from .balance_site import site_from
from .provenance import REPO_ROOT, header_line, provenance

DECLARATION_PATH = REPO_ROOT / "declarations" / "taylor_park.toml"
SITE = site_from(DECLARATION_PATH)

RIDGWAY = wb.SITE          # the first site, for the comparisons this report draws


def reporting_split(site=SITE) -> dict:
    """The closure residual split by whether every gauge reported that day.

    This analysis exists here and not in the Ridgway study because Ridgway has nothing to
    split: every one of its series is complete. Where gauges are intermittent the split is the
    difference between an imbalance the evidence shows and one the evidence's absence creates,
    and nothing else in the report separates them.
    """
    bs = wb.load(site.storage_sigma_base, site=site)
    Y = np.array([o.y for o in bs.observations], dtype=float)
    M = np.array([o.mask for o in bs.observations], dtype=bool)
    inflow = np.nansum(Y[:, 2:], axis=1)
    r = np.diff(Y[:, 0]) - CFS_DAY_TO_ACRE_FT * (inflow - Y[:, 1])[:-1]
    gauged_in = CFS_DAY_TO_ACRE_FT * inflow[:-1]
    full = M.all(axis=1)[:-1]
    finite = np.isfinite(r)

    def cell(sel) -> dict:
        v, g = r[sel], gauged_in[sel]
        return {
            "n_days": int(sel.sum()),
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "cumulative": float(v.sum()),
            "gauged_inflow_volume": float(np.nansum(g)),
            "residual_over_gauged_inflow": float(v.sum() / np.nansum(g)),
        }

    return {
        "units": f"{site.volume_unit} per day",
        "all_days": cell(finite),
        "every_gauge_reporting": cell(full & finite),
        "a_gauge_absent": cell(~full & finite),
        "ungauged_drainage_fraction": float(1.0 - site.gauged_area / site.total_area),
    }


def compute() -> dict:
    out = wb.compute(site=SITE)
    out["schema_version"] = "fsre-real-taylor-park-v1"
    # merge, never replace: compute() already put this site's series and drainage areas here
    out["site"] |= {"key": SITE.key, "label": SITE.label,
                    "declaration": str(DECLARATION_PATH.relative_to(REPO_ROOT))}
    out["provenance"]["generation"] = provenance()
    out["reporting_split"] = reporting_split()
    out["comparison_with_ridgway"] = _comparison()
    out["second_site_cost"] = {
        "declaration_only": False,
        "what_the_declaration_covered": [
            "states, units and both constraint variants",
            "five sensors with their selectors, declared sigmas and citations",
            "the committed record, the drainage areas, the process-noise scales and prior widths",
        ],
        "what_it_did_not_cover": [
            "estimators_balance assumed exactly two gauged inflows (N_SENSORS = 4 and three "
            "hand-indexed state layouts); the count now comes from BalanceConfig.n_inflows",
            "the study's functions read this module's globals rather than a declaration; "
            "experiments.balance_site.Site is the view that replaced them",
            "the closure and alignment residuals summed two named inflow columns; they now sum "
            "every inflow column the site declares",
        ],
        "paid_again_by_a_third_site": False,
    }
    out["claims"] = claims(out)
    return out


def _comparison() -> dict:
    """The same three numbers at both sites. Nothing is concluded from two points."""
    ridgway = json.loads((REPO_ROOT / "results" / "real_water_balance.json").read_text(encoding="utf-8"))
    rc = ridgway["closure_free"]
    return {
        "note": "Two sites is not a sample. These are reported because the second site's result "
                "is only interpretable against the first, not because two points support a law.",
        "ridgway": {
            "gauged_fraction": float(ridgway["site"]["gauged_fraction"]),
            "ungauged_drainage_fraction": float(1.0 - ridgway["site"]["gauged_fraction"]),
            "n_inflow_gauges": 2,
            "complete_series": True,
            "residual_over_gauged_inflow": float(rc["cumulative_as_fraction_of_gauged_inflow"]),
            "interval_contains_zero": bool(rc["cumulative_ci95_ar1"][0] <= 0.0 <= rc["cumulative_ci95_ar1"][1]),
        },
        "taylor_park": {
            "gauged_fraction": float(SITE.gauged_area / SITE.total_area),
            "ungauged_drainage_fraction": float(1.0 - SITE.gauged_area / SITE.total_area),
            "n_inflow_gauges": SITE.n_inflows,
            "complete_series": False,
        },
    }


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition.

    None of Ridgway's claims are reused. The shared claims() failed on this record, which is
    what it is for; these are the sentences that are true of THIS site.
    """
    split = r["reporting_split"]
    full, absent, every = split["every_gauge_reporting"], split["a_gauge_absent"], split["all_days"]
    cmp = r["comparison_with_ridgway"]
    missing = r["record"]["n_missing"]
    seasonal = sorted(missing.values())[-2:]
    return {
        "two_of_the_five_series_are_substantially_incomplete":
            all(n > 0.3 * r["record"]["n_days"] for n in seasonal),
        "ridgways_series_are_complete_and_this_sites_are_not":
            cmp["ridgway"]["complete_series"] and not cmp["taylor_park"]["complete_series"],
        "this_site_carries_more_inflow_gauges_than_the_first":
            cmp["taylor_park"]["n_inflow_gauges"] > cmp["ridgway"]["n_inflow_gauges"],
        "the_balance_does_not_close_here": every["residual_over_gauged_inflow"] > 0.05,
        "ridgways_balance_does_close": abs(cmp["ridgway"]["residual_over_gauged_inflow"]) < 0.01,
        "the_imbalance_survives_on_days_when_every_gauge_reports":
            full["residual_over_gauged_inflow"] > 0.05,
        "absent_readings_account_for_part_of_it_and_not_all":
            absent["mean"] > full["mean"] > 0.0,
        "the_split_leaves_less_than_all_of_the_imbalance_to_the_absent_days":
            full["cumulative"] > 0.3 * every["cumulative"],
        "the_full_reporting_residual_is_close_to_this_sites_ungauged_fraction":
            abs(full["residual_over_gauged_inflow"] - split["ungauged_drainage_fraction"]) < 0.02,
        "the_same_correspondence_fails_at_ridgway":
            abs(cmp["ridgway"]["residual_over_gauged_inflow"]
                - cmp["ridgway"]["ungauged_drainage_fraction"]) > 0.05,
        "the_second_site_was_not_a_declaration_alone":
            r["second_site_cost"]["declaration_only"] is False
            and len(r["second_site_cost"]["what_it_did_not_cover"]) >= 2,
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    split = report["reporting_split"]
    full, absent, every = split["every_gauge_reporting"], split["a_gauge_absent"], split["all_days"]
    cmp = report["comparison_with_ridgway"]
    cost = report["second_site_cost"]
    rec = report["record"]

    A("# A second reservoir, and what a second reservoir cost")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(f"{report['site']['label']}, from `{report['site']['declaration']}`. Three water years of "
      f"USGS daily values on the same grid as Ridgway: reservoir storage, the outlet gauge and "
      f"**three** inflow gauges.")
    A("")

    A("## What the second site cost")
    A("")
    A("When Ridgway's constants moved into a declaration, the claim was that a second reservoir "
      "would cost a declaration rather than a module. That was a hypothesis, and this is its "
      "test. The answer is a declaration **plus two one-time costs that only a second site "
      "could have exposed**:")
    A("")
    for line in cost["what_it_did_not_cover"]:
        A(f"- {line}")
    A("")
    A("The declaration itself needed nothing new — same schema, no added fields. What it could "
      "not cover was code that had quietly encoded the first site's shape. No test could have "
      "found it: every test had two inflow gauges, because every site did. None of it is paid "
      "again by a third site.")
    A("")

    A("## What the evidence looks like here")
    A("")
    A(f"| series | missing days of {rec['n_days']:,} |")
    A("| --- | ---: |")
    for source_id, n in rec["n_missing"].items():
        A(f"| `{source_id}` | {n:,} |")
    A("")
    A("Two of the three inflow gauges are seasonal: they report for part of each year only. "
      "Every Ridgway series is complete, every day, so **the bridge's missing-reading path had "
      "never been exercised by real evidence** — only by synthetic tests. Here it carries "
      f"{max(rec['n_missing'].values()):,} absent days on each of two columns.")
    A("")

    A("## The balance does not close, and the absent readings are not the whole reason")
    A("")
    A("Ridgway's three-year cumulative residual is indistinguishable from zero. This one is "
      "not, and the study's shared `claims()` **refused to print Ridgway's sentences about "
      "it** — the guard working on a site transfer rather than on a wording slip.")
    A("")
    A("Where gauges are intermittent, an imbalance the evidence shows and an imbalance the "
      "evidence's absence creates would otherwise be confounded. This record can separate them:")
    A("")
    A(f"| days | n | mean ({split['units']}) | cumulative | residual / gauged inflow |")
    A("| --- | ---: | ---: | ---: | ---: |")
    for name, cell in (("every gauge reporting", full), ("a seasonal gauge absent", absent),
                       ("all days", every)):
        A(f"| {name} | {cell['n_days']:,} | {cell['mean']:.2f} | {cell['cumulative']:,.0f} | "
          f"{cell['residual_over_gauged_inflow'] * 100:.2f}% |")
    A("")
    A(f"The last column divides by gauged inflow **as measured on those same days**, so the "
      f"middle row's {absent['residual_over_gauged_inflow'] * 100:.2f}% is not an imbalance of "
      f"that size: on a day a seasonal creek is absent it is missing from the denominator too, "
      f"and the remaining gauges are at their winter low. It is reported because the absence is "
      f"the point, not because the ratio is comparable to the row above it.")
    A("")
    A(f"So the absent readings account for part of the apparent imbalance and not all of it. On "
      f"the {full['n_days']:,} days when **every gauge reports**, storage still rises "
      f"{full['mean']:.2f} {split['units']} more than the gauges account for — "
      f"{full['residual_over_gauged_inflow'] * 100:.2f}% of gauged inflow. That is present in "
      f"fully observed data and this study does not say what it is.")
    A("")

    A("## One comparison, and the conclusion not drawn from it")
    A("")
    A("| | Ridgway | Taylor Park |")
    A("| --- | ---: | ---: |")
    A(f"| inflow gauges | {cmp['ridgway']['n_inflow_gauges']} | {cmp['taylor_park']['n_inflow_gauges']} |")
    A(f"| series complete | yes | no |")
    A(f"| ungauged drainage | {cmp['ridgway']['ungauged_drainage_fraction'] * 100:.2f}% | "
      f"{cmp['taylor_park']['ungauged_drainage_fraction'] * 100:.2f}% |")
    A(f"| residual / gauged inflow | {cmp['ridgway']['residual_over_gauged_inflow'] * 100:.2f}% | "
      f"{full['residual_over_gauged_inflow'] * 100:.2f}% (fully reported days) |")
    A("")
    A(f"At Taylor Park the residual on fully reported days, "
      f"{full['residual_over_gauged_inflow'] * 100:.2f}%, sits within "
      f"{abs(full['residual_over_gauged_inflow'] - split['ungauged_drainage_fraction']) * 100:.2f} "
      f"percentage points of its ungauged drainage fraction, "
      f"{split['ungauged_drainage_fraction'] * 100:.2f}%. It is the obvious explanation and the "
      f"augmented variant carries exactly that term.")
    A("")
    A(f"**It is not concluded here.** The same comparison at Ridgway is "
      f"{cmp['ridgway']['residual_over_gauged_inflow'] * 100:.2f}% against "
      f"{cmp['ridgway']['ungauged_drainage_fraction'] * 100:.2f}% ungauged — no correspondence "
      f"at all. A relationship that holds at one site and fails at the other is a coincidence "
      f"or a mechanism, and two sites cannot tell which. Gauge bias, the stage-capacity table, "
      f"the daily-mean alignment and real ungauged inflow all remain live, and nothing here "
      f"separates them.")
    A("")

    A("## What this does not establish")
    A("")
    for line in ("Two sites is not a sample. Nothing here is evidence about reservoirs in general.",
                 "No fault is diagnosed. A balance that does not close says the declared model and the "
                 "declared uncertainties disagree with the record; it does not say which is wrong.",
                 "The seasonal gauges' absence is treated as absence, not as zero flow. A creek that is "
                 "not reporting is not a creek that is not flowing, and the split above is what keeps "
                 "those apart rather than a correction that merges them.",
                 "The same daily-mean alignment approximation the Ridgway study documents applies here, "
                 "and is not re-derived.",
                 "The declared storage sigma is this consumer's, not USGS's, at both sites alike."):
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "real_taylor_park.md").write_text(text, encoding="utf-8")
    (out_dir / "real_taylor_park.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
