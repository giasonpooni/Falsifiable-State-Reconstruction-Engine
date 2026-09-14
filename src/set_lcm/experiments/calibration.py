"""In-loop null of the consistency statistic, a threshold x debounce sweep, and
the null of the evidence-side CUSUM channel.

The kernel test (tests/test_lcm.py) shows stat ~ chi2(1) for i.i.d. draws with
the TRUE covariance. That says nothing about the loop, where the statistic is
built from the filter's own reported P. This module measures that null
empirically over the windows where the joint hypothesis (constraint AND model
AND calibrated uncertainty) holds, and sweeps the guard's threshold and
debounce on the leak scenario so that false-alarm rate, detection and error can
be read off together instead of from one point.

The third section runs the unconstrained KF over the same nominal windows with
the per-sensor innovation CUSUM at decision intervals h in CUSUM_HS and counts,
per sensor, the alarms raised and the largest statistic reached, so the shipped
h can be read against the null rather than assumed.

    python -m set_lcm.experiments.calibration [n_seeds]   -> results/calibration.{md,json}
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..testbed.cusum import CusumConfig
from ..testbed.runner import EstimatorSpec
from .phase1 import DETECT_WITHIN, N_SEEDS, iter_runs, run_scenario

# Windows over which the joint hypothesis holds for the shipped scenarios.
NULL_WINDOWS: dict[str, tuple[int, int]] = {
    "closed_noise": (0, 600),
    "closed_blackout_pumpbias": (0, 50),
    "leak_stale_constraint": (0, 300),
    "bias_quant_delay": (0, 200),
}
THRESHOLDS = {"chi2_1(0.95)": 3.841, "chi2_1(0.99)": 6.635, "chi2_1(0.999)": 10.828}
NOMINAL = {"chi2_1(0.95)": 0.05, "chi2_1(0.99)": 0.01, "chi2_1(0.999)": 0.001}
KF = EstimatorSpec("kf", "kf", None)
SWEEP_QS = (0.95, 0.99, 0.999)
SWEEP_DEBOUNCES = (1, 3, 10)
CUSUM_HS = (4.0, 6.0, 8.0, 10.0)
CUSUM_SHIPPED = CusumConfig()          # the default on EstimatorSpec: k = 0.5, h = 8


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    den = float(x @ x)
    return float(x[:-1] @ x[1:]) / den if den > 0 else 0.0


def null_stats(n_seeds: int = N_SEEDS) -> dict:
    """Pool the unconstrained KF's consistency statistic over each null window and seed."""
    out = {}
    for name, (lo, hi) in NULL_WINDOWS.items():
        pooled, rhos = [], []
        for _, _, _, _, rr in iter_runs(name, KF, n_seeds):
            s = rr.stat[lo:hi]
            s = s[~np.isnan(s)]
            pooled.append(s)
            rhos.append(_lag1(s))
        s = np.concatenate(pooled)
        rho = float(np.mean(rhos))
        tau = (1.0 + rho) / (1.0 - rho) if rho < 1.0 else float("inf")   # AR(1) integrated autocorrelation time
        out[name] = {
            "window": [lo, hi],
            "n_seeds": n_seeds,
            "n_samples": int(s.size),
            "mean": float(s.mean()),
            "q95": float(np.quantile(s, 0.95)),
            "q99": float(np.quantile(s, 0.99)),
            "q999": float(np.quantile(s, 0.999)),
            "exceedance": {k: {"empirical": float((s > v).mean()), "nominal": NOMINAL[k]} for k, v in THRESHOLDS.items()},
            "lag1_autocorr": rho,
            "autocorr_time": tau,
            "effective_samples": float(s.size / tau) if np.isfinite(tau) else 0.0,
        }
    return out


def threshold_sweep(
    n_seeds: int = N_SEEDS,
    qs: tuple[float, ...] = SWEEP_QS,
    debounces: tuple[int, ...] = SWEEP_DEBOUNCES,
    scenario: str = "leak_stale_constraint",
    window: str = "leak_and_after",
) -> list[dict]:
    rows = []
    for q in qs:
        for d in debounces:
            spec = EstimatorSpec(f"kf+hard+guard(q={q},debounce={d})", "kf", "hard", guard=True, threshold_q=q, debounce=d)
            agg, _, _ = run_scenario(scenario, n_seeds, specs=[spec])
            a = agg[spec.name]
            rows.append({
                "q": q, "debounce": d,
                "fa_rate": a["false_alarms"]["rate"], "fa_max": a["false_alarms"]["max"],
                "detected_within": a["detection"]["detected_within"], "n": a["detection"]["n"],
                "median_delay": a["detection"]["median_delay"], "median_censored": a["detection"]["median_censored"],
                "rmse": a["rmse_by_window"][window]["mean"],
                "cov95": a["coverage95_by_window"][window]["mean"],
                "nz": a["nz_rms_by_window"][window]["mean"],
                "held": a["held_steps"]["mean"],
            })
    return rows


def cusum_null(n_seeds: int = N_SEEDS, hs: tuple[float, ...] = CUSUM_HS, k: float = CUSUM_SHIPPED.k) -> dict:
    """Null of the per-sensor innovation CUSUM: the unconstrained KF over each nominal
    window, once per decision interval h (the reset after an alarm makes the trajectory
    depend on h) and once with h = inf, whose max statistic is the largest un-reset
    excursion. Alarms and statistics are read on the report clock, exactly as the
    grid records them."""
    windows = {}
    for name, (lo, hi) in NULL_WINDOWS.items():
        by_h = {}
        for h in (*hs, float("inf")):
            spec = replace(KF, name=f"kf(cusum h={h:g})", cusum=CusumConfig(k=k, h=h))
            alarms = np.zeros(2, dtype=int)
            seeds_with_alarm = np.zeros(2, dtype=int)
            max_stat = np.zeros(2)
            for _, _, _, _, rr in iter_runs(name, spec, n_seeds):
                al = rr.cusum_alarm[lo:hi]
                st = rr.cusum_stat[lo:hi]
                alarms += al.sum(axis=0)
                seeds_with_alarm += al.any(axis=0)
                max_stat = np.maximum(max_stat, np.nanmax(st, axis=0))
            by_h[f"{h:g}"] = {"alarms": alarms.tolist(), "seeds_with_alarm": seeds_with_alarm.tolist(),
                              "max_stat": max_stat.tolist()}
        windows[name] = {"window": [lo, hi], "n_seeds": n_seeds, "n_steps_per_sensor": (hi - lo) * n_seeds, "by_h": by_h}

    total_steps = sum(w["n_steps_per_sensor"] for w in windows.values()) * 2
    totals = {f"{h:g}": int(sum(sum(w["by_h"][f"{h:g}"]["alarms"]) for w in windows.values())) for h in hs}
    zero = [h for h in hs if totals[f"{h:g}"] == 0]
    shipped = f"{CUSUM_SHIPPED.h:g}"
    return {
        "k": k, "hs": list(hs), "shipped_h": CUSUM_SHIPPED.h,
        "windows": windows,
        "total_alarms_by_h": totals,
        "total_sensor_steps": int(total_steps),
        "fa_rate_by_h": {hh: v / total_steps for hh, v in totals.items()},
        "smallest_h_zero_alarms": min(zero) if zero else None,
        "shipped_h_alarms": totals.get(shipped),
    }


def _render_cusum(cn: dict, n_seeds: int, sweep: list[dict]) -> list[str]:
    hs = [f"{h:g}" for h in cn["hs"]]
    shipped_guard = next((r for r in sweep if r["q"] == 0.999 and r["debounce"] == 3), None)
    md = ["# Null of the evidence-side CUSUM (kf, per sensor)", "",
          f"Two-sided CUSUM on the normalised innovation z = (y − H x_pred)/√Sᵢᵢ with k = {cn['k']:g}, "
          f"unconstrained KF, {n_seeds} seeds, over the same nominal windows. Cells are alarms s1 / s2 "
          "(seeds with at least one alarm s1 / s2) and, in the last column, the largest statistic either "
          "sensor reached with no reset (h = ∞). An alarm resets its channel, so the statistic and the "
          "alarm count at a given h come from a run at that h, not from thresholding one trace.", "",
          "| window | sensor-steps | " + " | ".join(f"h = {h}" for h in hs) + " | max stat (h = ∞) s1 / s2 |",
          "| --- | --- | " + " | ".join("---" for _ in hs) + " | --- |"]
    for name, w in cn["windows"].items():
        cells = []
        for h in hs:
            r = w["by_h"][h]
            cells.append(f"{r['alarms'][0]} / {r['alarms'][1]} ({r['seeds_with_alarm'][0]} / {r['seeds_with_alarm'][1]})")
        inf = w["by_h"]["inf"]["max_stat"]
        md.append(f"| {name} {w['window']} | {w['n_steps_per_sensor']} | " + " | ".join(cells) +
                  f" | {inf[0]:.2f} / {inf[1]:.2f} |")
    totals = cn["total_alarms_by_h"]
    rates = cn["fa_rate_by_h"]
    md += ["", "Totals over all nominal windows, both sensors (" + f"{cn['total_sensor_steps']} sensor-steps): " +
           ", ".join(f"h = {h}: {totals[h]} alarm{'s' if totals[h] != 1 else ''} ({rates[h]:.1e} per sensor-step)"
                     for h in hs) + "."]
    smallest = cn["smallest_h_zero_alarms"]
    shipped = f"{cn['shipped_h']:g}"
    if smallest is None:
        md.append(f"No h in {{{', '.join(hs)}}} gives zero alarms over all nominal windows and {n_seeds} seeds.")
    else:
        md.append(f"The smallest h in {{{', '.join(hs)}}} with zero alarms over all nominal windows and "
                  f"{n_seeds} seeds is h = {smallest:g}.")
    inf_max = max(max(w["by_h"]["inf"]["max_stat"]) for w in cn["windows"].values())
    guard_rate = "" if shipped_guard is None else (
        f", against {shipped_guard['fa_rate']:.1e} per step for the constraint guard at its shipped q = 0.999, "
        "debounce 3 (sweep above)")
    n_alarms = cn["shipped_h_alarms"]
    if n_alarms == 0:
        md.append(f"The shipped default h = {shipped} is kept: it produced no alarm on any nominal window. "
                  f"The largest un-reset excursion was {inf_max:.2f}, so zero alarms here is a statement about "
                  f"this sample, not a bound.")
    else:
        md.append(f"The shipped default h = {shipped} is kept: its {n_alarms} nominal alarm{'s' if n_alarms != 1 else ''} "
                  f"{'are' if n_alarms != 1 else 'is'} {rates[shipped]:.1e} per sensor-step{guard_rate}, and the grid's 'CUSUM FA max' column "
                  f"reports the per-sensor count in every scenario so they are never hidden. "
                  f"A larger h buys silence on this sample — the largest un-reset excursion was {inf_max:.2f} — "
                  f"at a delay cost of (h − {shipped}) / (z̄ − k) steps for a sustained shift z̄, i.e. a few "
                  f"steps for the 3 kg bias and more for the slowly building leak lag; zero alarms at a larger h "
                  f"is a statement about this sample, not a bound.")
    return md


def render(nulls: dict, sweep: list[dict], n_seeds: int, cusum: dict | None = None) -> str:
    md = ["# In-loop null of the consistency statistic", "",
          f"Unconstrained KF, {n_seeds} seeds, statistic r²/(A P Aᵀ) computed from the filter's own "
          "reported P over windows where the joint hypothesis holds. Under an exact χ²(1) null the mean "
          "would be 1.0 and the exceedances would equal the nominal tail probabilities. The lag-1 "
          "autocorrelation gives an AR(1) integrated autocorrelation time τ; the number of effectively "
          "independent samples is n/τ.", "",
          "| window | n | mean | q95 | q99 | q999 | P(>3.841) [0.05] | P(>6.635) [0.01] | P(>10.828) [0.001] | ρ₁ | τ | n_eff |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in nulls.items():
        e = v["exceedance"]
        md.append(f"| {name} {v['window']} | {v['n_samples']} | {v['mean']:.3f} | {v['q95']:.2f} | {v['q99']:.2f} | "
                  f"{v['q999']:.2f} | {e['chi2_1(0.95)']['empirical']:.4f} | {e['chi2_1(0.99)']['empirical']:.4f} | "
                  f"{e['chi2_1(0.999)']['empirical']:.5f} | {v['lag1_autocorr']:.3f} | {v['autocorr_time']:.1f} | "
                  f"{v['effective_samples']:.0f} |")
    md += ["", "# Threshold × debounce sweep (kf+hard+guard, leak_stale_constraint)", "",
           f"Pre-onset false-alarm rate per step, seeds detected within {DETECT_WITHIN} steps of onset, censored "
           "median delay, and (RMSE, cov95, nz) over the leak window. The shipped setting is q = 0.999, "
           "debounce = 3.", "",
           "| q | debounce | FA rate | FA max | detected | median delay | RMSE leak | cov95 leak | nz leak | held |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sweep:
        med = f"> {r['median_delay']:.0f}" if r["median_censored"] else f"{r['median_delay']:.0f}"
        md.append(f"| {r['q']} | {r['debounce']} | {r['fa_rate']:.1e} | {r['fa_max']} | {r['detected_within']}/{r['n']} | "
                  f"{med} | {r['rmse']:.2f} | {r['cov95']:.2f} | {r['nz']:.2f} | {r['held']:.0f} |")
    if cusum is not None:
        md += ["", *_render_cusum(cusum, n_seeds, sweep)]
    return "\n".join(md) + "\n"


def main(out_dir: Path, n_seeds: int = N_SEEDS, *, quiet: bool = False) -> int:
    from .provenance import header_line, provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    prov = provenance()
    nulls = null_stats(n_seeds)
    sweep = threshold_sweep(n_seeds)
    cusum = cusum_null(n_seeds)
    text = render(nulls, sweep, n_seeds, cusum).replace("\n\n", "\n\n" + header_line(prov) + "\n\n", 1)
    (out_dir / "calibration.md").write_text(text, encoding="utf-8")
    (out_dir / "calibration.json").write_text(
        json.dumps({"schema_version": "fsre-calibration-v1", "n_seeds": n_seeds, "provenance": prov, "null": nulls, "threshold_sweep": sweep,
                    "cusum_null": cusum}, indent=2),
        encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path.cwd() / "results", int(sys.argv[1]) if len(sys.argv) > 1 else N_SEEDS))
