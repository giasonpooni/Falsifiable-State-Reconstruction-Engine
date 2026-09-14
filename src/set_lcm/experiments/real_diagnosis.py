"""The diagnostic surface, pointed at real evidence for the first time.

`diagnose()` is the most developed decision machinery in this repository: candidate fits with
amplitude intervals, a nuisance space, a minimum separation between candidates, and explicit
`ambiguous` / `insufficient_evidence` outcomes. Until now every record it judged was one this
repository generated -- muskingum_reach, fluid_baseline and camera_baseline are synthetic, and
real_fluid_baseline calls it with an EMPTY hypothesis set, as a consistency test with no
candidates at all. The isolation half had never met real evidence.

results/real_taylor_park produced what was missing: a real record whose balance does not close,
by a margin that is not arguable. This asks the obvious next question of it -- what could
explain that? -- with a catalogue declared in the sites' own TOML rather than written here.

WHAT IS BEING DIAGNOSED. The daily closure residual r_k = (S_{k+1} - S_k) - c (inflows -
outflow)_k, summed into calendar-length windows. Aggregation is a declared choice with a
reason: the daily-mean alignment the Ridgway study documents is approximate at daily
resolution and averages down over a month, and a 36-long residual keeps the whitening exact
and quick where a 1,095-long one would not.

WHERE THE COVARIANCE COMES FROM. The declared instrument sigmas, propagated: a window's
residual telescopes to (S_end - S_start) minus c times the window's net flow, so its variance
is the two storage readings' declared variances plus c^2 times the summed flow variances.
Adjacent windows share a storage reading with opposite signs, which puts -sigma_S^2 on the
first off-diagonal. Nothing is fitted to the residual's own scatter, which would be circular.

THE ONE THING THAT IS SWEPT, AND WHY. Those declared sigmas describe instruments. They say
nothing about the daily-mean alignment error, which this repository has documented from the
start as present and unquantified. Rather than pick a value -- tuning, and forbidden here --
the report declares a per-day alignment sd and sweeps it, then says what the diagnosis is at
each point. The reader sees how much unmodelled error must be admitted before the record is
explicable at all, and what the engine will and will not name once it is.

WHAT TO EXPECT, WRITTEN DOWN BEFORE READING THE TABLE. A single closure row is rank 1, and the
repository's own bound says a rank-1 constraint cannot isolate: every fault it can see is
collinear with every other. `ungauged_constant` and `outflow_reads_low` are declared separately
because a practitioner thinks of them separately, and on this residual they are exactly the
same direction. A catalogue that named only those two could never resolve. The gaps in this
site's record are what make anything separable at all.

    python -m set_lcm.experiments.real_diagnosis  -> results/real_diagnosis.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..diagnostics import diagnose
from ..testbed.estimators_balance import CFS_DAY_TO_ACRE_FT
from . import real_water_balance as wb
from .balance_site import Site, site_from
from .provenance import REPO_ROOT, header_line, provenance

WINDOWS = 36                    # calendar-length windows over three water years
ALPHA = 0.01
INTERVAL_LEVEL = 0.95

# Per-day sd of the unmodelled daily-mean alignment error, in the sites' volume unit. Swept
# rather than chosen: the declared instrument sigmas do not describe it, and picking the value
# that makes a fit work is the thing this repository exists to refuse.
ALIGNMENT_SWEEP = (0.0, 50.0, 200.0, 500.0, 1000.0)

SITES = ("ridgway", "taylor_park")


def _record(site: Site) -> dict:
    """The residual, the window geometry and the declared variances, once per site."""
    bs = wb.load(site.storage_sigma_base, site=site)
    Y = np.array([o.y for o in bs.observations], dtype=float)
    mask = np.array([o.mask for o in bs.observations], dtype=bool)
    var = np.array([np.diag(o.R) for o in bs.observations], dtype=float)
    n = len(Y)
    edges = np.linspace(0, n - 1, WINDOWS + 1).astype(int)

    storage, outflow, inflows = Y[:, 0], Y[:, 1], Y[:, 2:]
    inflow_total = np.nansum(inflows, axis=1)
    daily = np.diff(storage) - CFS_DAY_TO_ACRE_FT * (inflow_total - outflow)[:-1]
    # A day is "a seasonal gauge absent" when some inflow column did not report. At a site whose
    # every series is complete this is zero throughout, and the candidate that rests on it has a
    # zero signature -- which diagnose() reports as unidentifiable rather than quietly fitting.
    absent = (~mask[:, 2:].all(axis=1))[:-1]

    var_storage = var[:, 0]
    var_net = (np.where(mask[:, 1], var[:, 1], 0.0)
               + np.nansum(np.where(mask[:, 2:], var[:, 2:], 0.0), axis=1))

    windows = []
    for a, b in zip(edges, edges[1:]):
        windows.append({
            "days": int(b - a),
            "residual": float(np.nansum(daily[a:b])),
            "storage_change": float(storage[b] - storage[a]),
            "absent_days": int(absent[a:b].sum()),
            "inflow_cfs_days": float(np.nansum(inflow_total[a:b])),
            "var_declared": float(var_storage[b] + var_storage[a]
                                  + CFS_DAY_TO_ACRE_FT ** 2 * np.nansum(var_net[a:b])),
            "var_shared_storage": float(var_storage[b]),
        })
    return {"windows": windows, "n_days": int(n)}


def covariance(rec: dict, alignment_sd: float) -> np.ndarray:
    """Declared instrument variance, plus the swept alignment term, with the off-diagonal the
    shared storage reading puts there."""
    w = rec["windows"]
    m = len(w)
    cov = np.zeros((m, m))
    for i, cell in enumerate(w):
        cov[i, i] = cell["var_declared"] + cell["days"] * alignment_sd ** 2
    for i in range(m - 1):
        # window i ends on the storage reading window i+1 starts on, with the opposite sign
        cov[i, i + 1] = cov[i + 1, i] = -w[i]["var_shared_storage"]
    return cov


PROFILES = {
    "constant_volume_per_day": lambda w: float(w["days"]),
    "constant_outflow_bias": lambda w: -CFS_DAY_TO_ACRE_FT * w["days"],
    "unreported_seasonal_inflow": lambda w: CFS_DAY_TO_ACRE_FT * w["absent_days"],
    "storage_scale_error": lambda w: w["storage_change"],
    "inflow_rating_scale_error": lambda w: -CFS_DAY_TO_ACRE_FT * w["inflow_cfs_days"],
}


def signatures(site: Site, rec: dict) -> dict[str, np.ndarray]:
    """Resolve each declared fault's profile against this record.

    The declaration names a profile and this owns the vocabulary, so an unknown name raises
    here with the known ones listed rather than being skipped.
    """
    out = {}
    for fault in site.decl.faults:
        if fault.profile not in PROFILES:
            raise ValueError(
                f"{site.key!r} declares fault {fault.name!r} with profile {fault.profile!r}, "
                f"which this study cannot resolve; it knows {sorted(PROFILES)}")
        out[fault.name] = np.array([PROFILES[fault.profile](w) for w in rec["windows"]])
    return out


def _cosines(sig: dict[str, np.ndarray]) -> list[dict]:
    """Pairwise geometry BEFORE any covariance, because exact collinearity survives all of them."""
    names = list(sig)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            u, v = sig[a], sig[b]
            na, nb = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            cos = float(u @ v / (na * nb)) if na > 0 and nb > 0 else None
            pairs.append({
                "a": a, "b": b, "cos": cos,
                "exactly_collinear": cos is not None and abs(abs(cos) - 1.0) < 1e-12,
                "zero_signature": na == 0.0 or nb == 0.0,
            })
    return pairs


def diagnose_site(key: str) -> dict:
    site = site_from(REPO_ROOT / "declarations" / f"{key}.toml")
    rec = _record(site)
    sig = signatures(site, rec)
    residual = np.array([w["residual"] for w in rec["windows"]])

    sweep = []
    for alignment_sd in ALIGNMENT_SWEEP:
        d = diagnose(residual, covariance(rec, alignment_sd), sig,
                     alpha=ALPHA, interval_level=INTERVAL_LEVEL)
        sweep.append({
            "alignment_sd": alignment_sd,
            "status": d.status,
            "null_statistic": d.null_statistic,
            "null_dof": d.null_dof,
            "null_threshold": d.null_threshold,
            "min_separation": d.min_separation,
            "adequate": [f.name for f in d.fits if f.adequate],
            "fits": [{"name": f.name, "observable": f.observable, "adequate": f.adequate,
                      "fit_statistic": f.fit_statistic, "amplitude": f.amplitude,
                      "amplitude_sd": f.amplitude_sd, "interval": list(f.interval) if f.interval else None,
                      "nearest": f.nearest, "nearest_cos": f.nearest_cos,
                      "nearest_orthogonal_fraction": f.nearest_orthogonal_fraction,
                      "nearest_isolation_amplification": f.nearest_isolation_amplification,
                      "amplitude_unit": site.decl.fault(f.name).amplitude_unit}
                     for f in d.fits],
            "explanation": d.explanation,
        })
    return {
        "key": site.key,
        "label": site.label,
        "n_days": rec["n_days"],
        "windows": len(rec["windows"]),
        "cumulative_residual": float(residual.sum()),
        "catalogue": [{"name": f.name, "label": f.label, "profile": f.profile,
                       "amplitude_unit": f.amplitude_unit, "description": f.description}
                      for f in site.decl.faults],
        "geometry": _cosines(sig),
        "sweep": sweep,
    }


def compute() -> dict:
    out = {
        "schema_version": "fsre-real-diagnosis-v1",
        "scope": "The diagnostic surface applied to real records, with the candidate catalogue "
                 "declared in each site's own TOML. No fault is asserted to exist at either site.",
        "provenance": {"generation": provenance()},
        "declared": {
            "windows": WINDOWS,
            "alpha": ALPHA,
            "interval_level": INTERVAL_LEVEL,
            "alignment_sd_swept": list(ALIGNMENT_SWEEP),
            "alignment_note":
                "per-day sd of the unmodelled daily-mean alignment error, in acre-ft. The declared "
                "instrument sigmas describe instruments and say nothing about it; it is swept rather "
                "than chosen because choosing the value that makes a fit work is tuning.",
            "covariance_source":
                "the declared instrument sigmas, propagated through the window sum, with the "
                "off-diagonal the shared storage reading puts there. Nothing is fitted to the "
                "residual's own scatter.",
            "profiles_known_to_this_study": sorted(PROFILES),
        },
        "sites": {key: diagnose_site(key) for key in SITES},
        "limitations": [
            "No fault is asserted to exist at either site. A candidate being adequate means the record does not reject it, which is not evidence that it happened.",
            "The catalogue is what each site declares. A cause absent from it cannot be found, and the ambiguity reported here is a property of the catalogue and the constraint, not a property of the reservoir.",
            "One closure row is rank 1, so its residual cannot separate candidates that act on it identically. That is a structural bound, not a limitation of the data.",
            "The alignment term is swept, not measured. Nothing here establishes its true size; the report says what the diagnosis is at each declared value and no more.",
            "Windows are equal-length divisions of the record, not calendar months, so seasonality is approximate at the window edges.",
            "Amplitude intervals are conditional on the single-candidate model and on the declared covariance at that sweep point.",
        ],
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition."""
    tp, rg = r["sites"]["taylor_park"], r["sites"]["ridgway"]

    def at(site, sd):
        return next(c for c in site["sweep"] if c["alignment_sd"] == sd)

    collinear = [p for p in tp["geometry"] if p["exactly_collinear"]]
    statuses = [c["status"] for c in tp["sweep"]]
    return {
        "the_two_constant_rate_candidates_are_exactly_collinear_on_this_residual": any(
            {p["a"], p["b"]} == {"ungauged_constant", "outflow_reads_low"} for p in collinear),
        "that_collinearity_holds_at_both_sites": any(
            {p["a"], p["b"]} == {"ungauged_constant", "outflow_reads_low"}
            for p in rg["geometry"] if p["exactly_collinear"]),
        "the_declared_instrument_sigmas_alone_explain_neither_site": all(
            at(s, 0.0)["status"] == "unexplained" for s in (tp, rg)),
        "the_second_sites_record_is_far_more_inconsistent_than_the_first_at_every_sweep_point": all(
            a["null_statistic"] > b["null_statistic"]
            for a, b in zip(tp["sweep"], rg["sweep"])),
        "the_engine_never_names_a_single_cause_at_the_second_site":
            "identified" not in statuses,
        "there_is_a_declared_alignment_at_which_it_reports_ambiguity": "ambiguous" in statuses,
        "admitting_enough_alignment_error_makes_the_record_consistent_with_no_fault":
            tp["sweep"][-1]["status"] == "consistent",
        "the_status_only_ever_loosens_as_more_error_is_admitted": all(
            a["null_statistic"] >= b["null_statistic"]
            for a, b in zip(tp["sweep"], tp["sweep"][1:])),
        "the_separable_candidate_at_the_second_site_is_the_one_its_gaps_create": any(
            p["cos"] is not None and abs(p["cos"]) < 0.75
            for p in tp["geometry"]
            if "seasonal_creeks_unreported" in (p["a"], p["b"])),
        "the_first_sites_catalogue_declares_no_seasonal_candidate_because_it_has_no_gaps":
            all(f["name"] != "seasonal_creeks_unreported" for f in rg["catalogue"]),
        "every_declared_fault_resolved_to_a_signature": all(
            len(s["geometry"]) == len(s["catalogue"]) * (len(s["catalogue"]) - 1) // 2
            for s in (tp, rg)),
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    d = report["declared"]
    tp, rg = report["sites"]["taylor_park"], report["sites"]["ridgway"]

    A("# What the diagnostic surface says about a real record")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A("`diagnose()` is the most developed decision machinery here: candidate fits with amplitude "
      "intervals, a nuisance space, a minimum separation between candidates, and explicit "
      "`ambiguous` and `insufficient_evidence` outcomes. Every record it had judged until now "
      "was one this repository generated. `real_fluid_baseline` calls it on a real record but "
      "with an **empty** hypothesis set — a consistency test with no candidates at all. The "
      "isolation half had never met real evidence.")
    A("")
    A("`results/real_taylor_park` produced what was missing: a real record whose balance does "
      "not close, by a margin that is not arguable. This asks the obvious next question of it.")
    A("")

    A("## The catalogue is declared, not written here")
    A("")
    A("Each site's candidates live in its own TOML. A fault's *signature* cannot: it is what "
      "the candidate would do to this particular residual, so it depends on how many days each "
      "window held and what the gauges saw. The declaration names a profile and this study "
      "resolves it — the same split as a constraint row whose right-hand side is a reading.")
    A("")
    for site in (tp, rg):
        A(f"**{site['label']}** — {len(site['catalogue'])} candidates:")
        A("")
        A("| candidate | amplitude unit | what it is |")
        A("| --- | --- | --- |")
        for f in site["catalogue"]:
            A(f"| `{f['name']}` | {f['amplitude_unit']} | {f['label']} |")
        A("")
    A(f"The first site declares no seasonal candidate because it has no gaps for one to live "
      f"in: every Ridgway series is complete. That difference is not a convenience — it is what "
      f"makes anything separable at the second site at all, as the next table shows.")
    A("")

    A("## The structural result, which no covariance can change")
    A("")
    A("Two candidates a practitioner thinks of as entirely different things — water no gauge "
      "sees, and an outlet gauge reading low — do the same thing to a closure residual: they "
      "add a constant volume per day. They are declared separately because they *are* different "
      "physically, and on this residual they are **exactly the same direction**.")
    A("")
    A("| site | pair | cos |")
    A("| --- | --- | ---: |")
    for site in (tp, rg):
        for p in site["geometry"]:
            if p["exactly_collinear"]:
                A(f"| {site['key']} | `{p['a']}` vs `{p['b']}` | {p['cos']:+.4f} |")
    A("")
    A("This is the rank-1 bound arriving in a real catalogue. One closure row gives a scalar per "
      "window, so every pair of candidates it can see at all is collinear in that one direction; "
      "no covariance, no amount of data and no threshold choice separates them. A catalogue "
      "holding only those two could never resolve, at any site, ever. Reporting that is the "
      "engine working — the alternative is naming one of them and being right half the time.")
    A("")
    A(f"What breaks the tie at {tp['label'].split(',')[0]} is the record's own gaps. "
      f"`seasonal_creeks_unreported` follows the count of absent days rather than the calendar, "
      f"so it points somewhere else:")
    A("")
    A("| pair | cos |")
    A("| --- | ---: |")
    for p in tp["geometry"]:
        if "seasonal_creeks_unreported" in (p["a"], p["b"]) and p["cos"] is not None:
            other = p["b"] if p["a"] == "seasonal_creeks_unreported" else p["a"]
            A(f"| `seasonal_creeks_unreported` vs `{other}` | {p['cos']:+.4f} |")
    A("")

    A("## What the engine says, against how much unmodelled error is admitted")
    A("")
    A("The covariance is the declared instrument sigmas, propagated through the window sum. "
      "Nothing is fitted to the residual's own scatter — that would be circular. But those "
      "sigmas describe *instruments*, and say nothing about the daily-mean alignment error this "
      "repository has documented as present and unquantified from the start. Picking a value "
      "for it would be tuning, so it is declared per day and swept.")
    A("")
    for site in (tp, rg):
        A(f"**{site['label']}** — cumulative residual {site['cumulative_residual']:,.0f} acre-ft "
          f"over {site['windows']} windows:")
        A("")
        A("| alignment sd (acre-ft/day) | null χ² | threshold | status | candidates not rejected |")
        A("| ---: | ---: | ---: | --- | ---: |")
        for c in site["sweep"]:
            A(f"| {c['alignment_sd']:,.0f} | {c['null_statistic']:,.1f} | "
              f"{c['null_threshold']:,.1f} | `{c['status']}` | {len(c['adequate'])} |")
        A("")
    A(f"Read the second site's column downward and the engine changes its mind in exactly the "
      f"way it should. With instrument sigmas alone the record is **unexplained** — not by one "
      f"candidate, by all of them; the declared uncertainty is far too tight for anything in "
      f"the catalogue to account for what the gauges did. Admit more unmodelled error and "
      f"several candidates become adequate *at once*, which is **ambiguous**: something is "
      f"there and the engine cannot tell you which. Admit enough and the record stops needing "
      f"an explanation at all — `consistent`, with nothing to diagnose.")
    A("")
    A("**At no point does it name a single cause.** That is the honest outcome for a rank-1 "
      "constraint and a catalogue containing two collinear members, and it was predicted before "
      "the table was computed. An engine that named one here would be wrong in a way that is "
      "hard to detect and expensive to trust.")
    A("")
    A(f"The first site sits below the second at every sweep point — its balance closes, so "
      f"there is less to explain — which is the same ordering its closure residual and its "
      f"ungauged estimate give, by the same evidence.")
    A("")

    A("## What this does not establish")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "real_diagnosis.md").write_text(text, encoding="utf-8")
    (out_dir / "real_diagnosis.json").write_text(
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
