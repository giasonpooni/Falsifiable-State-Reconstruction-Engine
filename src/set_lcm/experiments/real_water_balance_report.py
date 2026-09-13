"""Rendering for experiments.real_water_balance: the markdown the JSON is the record of.

Every number printed here comes from the result dict and nothing is recomputed, so the
report and results/real_water_balance.json cannot disagree.
"""
from __future__ import annotations


def _f(x, n=2):
    return "n/a" if x is None else f"{x:,.{n}f}"


def _ci(interval, n=0):
    """A two-element interval, or "n/a" when the result dict reports none."""
    if interval is None:
        return "n/a"
    lo, hi = interval
    return f"[{_f(lo, n)}, {_f(hi, n)}]"


def render_report(r: dict) -> str:
    site, dec, rec = r["site"], r["declared"], r["record"]
    c, al, blind = r["closure_free"], r["alignment"], r["blind"]
    L: list[str] = []
    A = L.append

    A("# P4b: a real reservoir water balance through the reconciliation kernel")
    A("")
    A(f"{site['reservoir']}, {rec['n_days']} days from {rec['first_day'][:10]}, four USGS daily-mean "
      f"series admitted by DAF. Gauged drainage {site['gauged_sq_mi']:.1f} of {site['total_sq_mi']:.0f} "
      f"sq mi = {site['gauged_fraction']:.3f}.")
    A("")
    A("**Truth-free.** Nobody knows how much water is in this reservoir, so no number below is an "
      "error. Every number is computed from the readings, or from a run's own record.")
    A("")
    A("This is the first constraint in the repository that is a physical law over evidence from "
      "independent instruments, rather than a declared relation in a simulator. It is **expected to "
      "fail**: about 7% of the catchment is ungauged, and evaporation, precipitation on the lake and "
      "any gauge bias land in the same unmeasured term. The question is whether it fails for the "
      "right reason and by the right amount.")
    A("")

    A("## The record")
    A("")
    A("| series | role | USGS | parameter / statistic | range |")
    A("|---|---|---|---|---|")
    for role, s in site["series"].items():
        if role == "storage":
            rng = f"{_f(rec['storage_acre_ft']['min'], 0)} – {_f(rec['storage_acre_ft']['max'], 0)} acre-ft"
        elif role == "outflow":
            rng = f"{_f(rec['outflow_cfs']['min'], 1)} – {_f(rec['outflow_cfs']['max'], 1)} ft³/s"
        else:
            rng = "see combined inflow below"
        A(f"| `{s['source_id']}` | {role.replace('_', ' ')} | {s['monitoring_location_id']} | "
          f"{s['parameter_code']} {s['unit']}, statistic {s['statistic_id']} | {rng} |")
    A("")
    A(f"Combined gauged inflow {_f(rec['inflow_cfs']['min'], 1)} – {_f(rec['inflow_cfs']['max'], 1)} ft³/s. "
      f"Missing readings per series: {rec['n_missing']}.")
    A("")

    A("## The closure residual, before filtering")
    A("")
    A("Arithmetic on the readings, without a filter or uncertainty weights. The daily-mean "
      "time pairing is an approximation, so even perfect gauges need not give zero:")
    A("")
    A("    r_k = (S_{k+1} − S_k) − c (q_in1 + q_in2 − q_out)_k,    "
      f"c = {dec['cfs_day_to_acre_ft']:.10f} acre-ft per ft³/s-day")
    A("")
    A(f"over {c['n_days']} days, in acre-ft per day:")
    A("")
    A("| mean | sd | median | 5th | 95th | min | max | lag-1 |")
    A("|---|---|---|---|---|---|---|---|")
    A(f"| {_f(c['mean'])} | {_f(c['sd'])} | {_f(c['median'])} | {_f(c['p05'])} | {_f(c['p95'])} | "
      f"{_f(c['min'])} | {_f(c['max'])} | {c['lag1_autocorr']:+.3f} |")
    A("")
    A(f"As a flow: mean **{c['mean_as_cfs']:+.3f} ft³/s**, sd {_f(c['sd_as_cfs'])} ft³/s, against a mean "
      f"gauged inflow of {_f(c['mean_throughput_cfs'], 1)} ft³/s. Over the whole record the imbalance "
      f"accumulates to {_f(c['cumulative'], 0)} acre-ft, "
      f"{c['cumulative_as_fraction_of_gauged_inflow'] * 100:+.2f}% of the "
      f"{_f(c['gauged_inflow_volume'], 0)} acre-ft that flowed in.")
    A("")
    A("**That cumulative is not evidence of a net imbalance, and must not be read as one.** It is a "
      f"sum of {c['n_days']:,} daily residuals, so it grows as sqrt(n) even when the gauges close "
      "exactly. Against its own standard error it is indistinguishable from zero:")
    A("")
    A("| the cumulative | standard error of the sum | in standard errors | 95% interval |")
    A("|---|---|---|---|")
    A(f"| {_f(c['cumulative'], 0)} acre-ft | {_f(c['cumulative_se'], 0)} independent, "
      f"{_f(c['cumulative_se_ar1'], 0)} under AR(1) | {_f(c['cumulative_in_se'])} independent, "
      f"{_f(c['cumulative_in_se_ar1'])} under AR(1) | {_ci(c['cumulative_ci95_ar1'])} |")
    A("")
    A(f"The interval contains zero, so **the three-year total is consistent with the gauges closing "
      f"exactly**; the AR(1) column uses the measured lag-1 of {c['lag1_autocorr']:+.3f} and is a model, "
      "not a measurement. What the record does show is a **daily** disagreement of tens of acre-feet, "
      "which is a different statement and the one the sections below test.")
    A("")

    A("### Which day's flow the storage change belongs with")
    A("")
    A("Storage here is a daily **mean**, so the pairing is not obvious. All three are computed:")
    A("")
    A("| pairing | mean (ft³/s) | sd (ft³/s) | lag-1 |")
    A("|---|---|---|---|")
    for name in ("same_day", "centred", "next_day"):
        a = al[name]
        A(f"| {name.replace('_', ' ')} | {a['mean_as_cfs']:+.3f} | {_f(a['sd_as_cfs'])} | "
          f"{a['lag1_autocorr']:+.3f} |")
    A("")
    A(f"The **{al['smallest_sd'].replace('_', ' ')}** pairing has the smallest scatter, which is what a "
      "change between two daily means should pair with. It is not proof: " + al["note"] + ".")
    A("")

    A("## What the constraint is blind to")
    A("")
    A(f"For the open constraint `A = {blind['A']}`, d(f) = fᵀAᵀ(A P Aᵀ + Σ_b)⁻¹A f per unit direction:")
    A("")
    A("| direction | d(f) |")
    A("|---|---|")
    for name, d in blind["d"].items():
        # Report the analytic null as such, rather than platform-specific SVD roundoff.
        # The raw computed d remains in JSON; no weak nonzero direction is rounded to null.
        shown = "0 (structural null)" if name == "null_space (S + G)" else f"{d:.6g}"
        A(f"| {name} | {shown} |")
    A("")
    A("- **Structurally invisible:** " + blind["structurally_invisible"]["meaning"])
    A("- **Not even a direction:** " + blind["not_even_a_direction"]["meaning"])
    A("")
    A(f"({blind['P_note']})")
    A("")

    A("## Through the kernel")
    A("")
    A("R is **consumer-declared** throughout: USGS states no per-value uncertainty anywhere in the "
      "daily-values API, so the bridge required a citation for every σ and carried it here.")
    A("")
    A(f"- Flows: σ = {dec['flow_sigma_relative'] * 100:.0f}% of the reading, floored at "
      f"{dec['flow_sigma_floor']:.1f} ft³/s. {dec['flow_sigma_citation']}")
    A(f"- Storage: **swept**, not declared once — {dec['storage_sigma_sweep']} acre-ft. "
      f"{dec['storage_sigma_citation']}")
    A(f"- The daily interval itself is an interpretation, not a source statement: "
      f"{dec['day_zone_citation']}")
    A("")
    A("Process noise, declared and never fitted: q_storage = "
      f"{dec['config']['q_storage']:.0f} acre-ft/√day, q_flow = {dec['config']['q_flow']:.0f} ft³/s/√day, "
      f"q_ungauged = {dec['config']['q_ungauged']:.0f} acre-ft/√day (wb_aug only).")
    A("")

    for key, s in r["sweep"].items():
        A(f"### Storage σ = {s['storage_sigma']:g} acre-ft")
        A("")
        A("| estimator | consistency stat mean | max | over threshold | flagged steps | |correction| mean | "
          "|residual| after |")
        A("|---|---|---|---|---|---|---|")
        for name, e in s["specs"].items():
            cs = e.get("consistency_stat")
            corr = e.get("correction_norm")
            post = e.get("residual_post")
            A(f"| `{name}` | {_f(cs['mean']) if cs else '—'} | {_f(cs['max']) if cs else '—'} | "
              f"{(f'{cs['fraction_over_threshold'] * 100:.1f}%' if cs and cs.get('fraction_over_threshold') is not None else '—')} | "
              f"{e['flagged_steps']} | {_f(corr['mean'], 1) if corr else '—'} | "
              f"{_f(post['mean_abs'], 1) if post else '—'} |")
        A("")
        thr = next((e["consistency_stat"]["threshold"] for e in s["specs"].values()
                    if e.get("consistency_stat")), None)
        if thr is not None:
            A(f"Threshold χ²(1) at q = 0.999 is {thr}. Under the declared hypothesis the statistic would "
              "have mean 1.")
        augs = [(n, e["ungauged_cumulative"]) for n, e in s["specs"].items() if "ungauged_cumulative" in e]
        if augs:
            A("")
            A("Cumulative ungauged net inflow U, acre-ft:")
            A("")
            A("| run | final | sd | in sd | min | max | as % of gauged inflow |")
            A("|---|---|---|---|---|---|---|")
            for n, a in augs:
                A(f"| `{n}` | {_f(a['final'], 0)} | {_f(a['final_sd'], 0)} | {_f(a['final_in_sd'])} | "
                  f"{_f(a['min'], 0)} | {_f(a['max'], 0)} | "
                  f"{a['final_as_fraction_of_gauged_inflow'] * 100:+.2f}% |")
            A("")
            A("**U is observed by nothing except the constraint.** In `wb_aug`, with no projection, it "
              "stays at its prior of 0 for the whole record while its sd grows — the augmented state is "
              "not identified by the data at all, unlike the simulated `kf_aug`, whose boundary flux L "
              "is identified through the dynamics. `+hard` moves the REPORTED U; only `+feedback` puts "
              "the projection back into the full filter state and covariance. A constraint fed back "
              "is absorbed as if it were fresh evidence; subsequent agreement is therefore not "
              "independent validation. That is also why `+feedback` carries the largest `in sd` "
              "above: feeding the constraint back shrinks the very sd that column divides by, so "
              "its apparent separation from zero is the least trustworthy of the three, not the "
              "most. The finite prior and process variance on U still permit rejection of "
              "sufficiently large disagreements on other records.")
            mid = dict(augs).get("wb_aug+hard")
            if mid is not None:
                A("")
                A(f"Cross-check: `wb_aug+hard` puts the three-year imbalance at {_f(mid['final'], 0)} "
                  f"acre-ft ({mid['final_as_fraction_of_gauged_inflow'] * 100:+.2f}% of gauged inflow), "
                  f"against {_f(c['cumulative'], 0)} acre-ft "
                  f"({c['cumulative_as_fraction_of_gauged_inflow'] * 100:+.2f}%) from the "
                  "arithmetic at the top of this report. These are two calculations from the same "
                  "measurements: a filter under declared noise, and a sum under a time-pairing assumption. "
                  "They are not required to agree: the filter's U is a smoothed quantity under a "
                  "declared random walk, and the arithmetic is not.")
                A("")
                A("**And their agreement is not evidence of an imbalance.** The filter reports its own "
                  f"sd of {_f(mid['final_sd'], 0)} acre-ft on that {_f(mid['final'], 0)} — "
                  f"{_f(mid['final_in_sd'])} sd from zero — and the arithmetic total is "
                  f"{_f(c['cumulative_in_se_ar1'])} standard errors from zero, an interval that "
                  "contains zero. Two numbers this uncertain landing near each other is agreement "
                  "between two uncertain numbers, not corroboration of a net imbalance.")
        A("")

    A("## What these numbers do not show")
    A("")
    for line in (
        "**That the constraint's rejection measures the ungauged flux.** The statistic rejects the "
        "declared closure; it does not say which of the ungauged catchment, evaporation, precipitation "
        "on the lake, the stage-capacity table or a gauge rating is responsible. That is the known "
        "limit of a global χ² test on constraint residuals (Crowe, 1985), and nothing here escapes it.",
        "**That R is calibrated.** Every R is a consumer declaration with a citation that says what it "
        "assumes. The flow σ assumes a 'Good' rating this repository did not acquire and a normal "
        "error; the storage σ has no source at all, which is why it is swept rather than declared.",
        "**That `wb_aug`'s U is the ungauged inflow.** U is whatever makes the balance close. It "
        "absorbs the ungauged catchment, evaporation, precipitation, the stage-capacity table's error "
        "and any gauge bias, in one number. Its sd is the filter's own, under declared noise, and is "
        "not an error bar on the physical quantity.",
        "**That `wb_closed` is worse or better.** It declares no constraint, so no statistic here can "
        "reject it. It is the baseline that shows what assuming closure looks like, not a candidate "
        "that lost.",
        "**Anything about an equal-and-opposite gauge bias.** It cancels before it reaches the state. "
        "The balance cannot see it. Individual channel predictions may still disagree, but that "
        "does not by itself establish which gauge is faulty.",
        "**That the reference is independent.** The first storage reading supplies b and also "
        "enters the estimate. The kernel does not include their shared-reference cross-covariance. "
        "Repeated feedback of this reference is not independent evidence.",
        "**That the alignment question is settled.** The centred pairing has the smallest scatter and "
        "also averages two readings, which would reduce the scatter regardless. Separating the two "
        "needs a sub-daily record, which is a different acquisition.",
        "**Generality.** One reservoir, three water years, one climate. Winter ice affects the inflow "
        "records here — a substantial number of daily values at these sites carry USGS's ESTIMATED "
        "qualifier, which DAF keeps out of content by design, so every reading is scored alike above "
        "and nothing joins the qualifier to the residual. Such qualifiers would be corroborating "
        "metadata, not ground-truth fault labels.",
    ):
        A(f"- {line}")
    A("")
    return "\n".join(L)
