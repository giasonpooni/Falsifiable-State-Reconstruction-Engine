"""What treating an uncertain relation as exact costs the ESTIMATE, not just the test.

results/errors_in_variables measured the first half of this. A constraint set may declare
A_var -- uncertainty on the relation's own coefficients -- and the consistency statistic and
detectability both widen S by Cov(E x) when it does. The projection did not. `project_hard`
and `project_soft` branched on `b_var` alone, so a set that declared its routing coefficients
uncertain was reconciled against A_bar as though the coefficients were exact, silently, while
the two functions immediately above them in the same module refused to answer without a state.
That was the gap the calibration study listed against itself. This closes it and prices it.

The change is one line of algebra. With A = A_bar + E the residual is

    r = A_bar x_hat - b = A_bar delta - E x_true - eps

and E is independent of the state error and zero-mean, so the cross-covariance between the
state error and the residual -- the numerator of the gain -- does not change at all:

    K = P A_bar^T S^-1        unchanged
    S = A_bar P A_bar^T + Sigma_b + Cov(E x)      widened

The gain therefore shrinks, and it shrinks by an amount that is a QUADRATIC FORM IN THE STATE.
That is the part no constant inflation of Sigma_b can imitate, and it is what this experiment
is built to test: the same constraint set, the same declared uncertainties, the operating
point scaled. If the correction is really quadratic in the state then treating A as exact must
get worse quadratically while the errors-in-variables update stays calibrated. If instead the
cost were some fixed mis-declaration, both would sit flat and the mechanism claimed here would
be wrong.

Scored on the estimate, against a null true by construction, by three measures:

    NEES        (x* - x_true)^T P*^-1 (x* - x_true), which a calibrated estimate puts at n
    coverage    how often that lands inside the nominal 95% ellipsoid
    RMS error   whether the estimate is actually better, not merely more honest about itself

The no-projection baseline is scored on the same draws, because the question a reader asks
next is whether an over-confident projection is worse than not projecting at all.

Truth-free this is not, and deliberately: the generating model is known because the point is
to measure a calibration against a null that is true by construction. It reuses the relation,
the state, the covariances and the declarations of results/errors_in_variables unchanged, so
the two artifacts describe the same system and their numbers can be read side by side. No real
record appears and nothing here says what any instrument does.

    python -m set_lcm.experiments.eiv_projection  -> results/eiv_projection.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..lcm import chi2_quantile, matrix_uncertainty, project_hard
from ..schema import ConstraintSet
from .errors_in_variables import A_BAR, B_VAR, P, STATE, declarations
from .provenance import REPO_ROOT, header_line, provenance

SEED = 20260914
DRAWS = 8000

# The rows' uncertain coefficients come from the same measured flow, so their errors move
# together. The middle of the calibration study's sweep: dependent, which is the realistic case.
SHARED = 0.01

# The operating point, scaled. Cov(E x) is quadratic in the state while Sigma_b and P are not,
# so this axis separates the mechanism from any fixed mis-declaration.
STATE_SCALE_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0)

NO_PROJECTION = "no projection"
EXACT = "A treated as exact"
FULL = "A_var, full vec(A) covariance"
COVERAGE_LEVEL = 0.95


def _nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """(x* - x_true)^T P*^-1 (x* - x_true). Calibrated means this averages the state dimension."""
    return float(error @ np.linalg.solve(covariance, error))


def _reprojected(x_hat: np.ndarray, cs: ConstraintSet, x_star: np.ndarray,
                 P_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The update the kernel would produce if it re-evaluated Cov(E x) at its own output.

    This is NOT the kernel. `project_hard` takes Cov(E x) at the incoming state and stops, one
    step, because the quadratic form has to be evaluated somewhere and the incoming state is
    the only one available before the update exists. Whether that choice costs anything is a
    question to measure, not to argue, so this recomputes the gain at the updated state and
    redoes the same update from the same prior. It is written out in full rather than calling
    a private kernel helper precisely because it is the comparison arm and not the shipped path.
    """
    A = np.atleast_2d(np.asarray(cs.A, dtype=float))
    b = np.asarray(cs.b, dtype=float).reshape(-1)
    Sigma = cs.b_cov + matrix_uncertainty(cs, x_star, P_star)
    S = A @ P @ A.T + Sigma
    K = P @ A.T @ np.linalg.inv(S)
    x_next = x_hat - K @ (A @ x_hat - b)
    I_KA = np.eye(P.shape[0]) - K @ A
    P_next = I_KA @ P @ I_KA.T + K @ Sigma @ K.T
    return x_next, 0.5 * (P_next + P_next.T)


def draw_projections(scale: float, draws: int = DRAWS, seed: int = SEED) -> dict:
    """One null experiment at one operating point, scored on the estimate."""
    declared = declarations(SHARED)
    truth = declared.pop("__truth__")
    n, m = A_BAR.shape[1], A_BAR.shape[0]
    x_true = STATE * scale
    rng = np.random.default_rng(seed)

    arms = [NO_PROJECTION, *declared]
    nees = {name: np.empty(draws) for name in arms}
    sq_error = {name: np.empty(draws) for name in arms}
    reported_var = {name: np.empty(draws) for name in arms}
    correction = {name: np.empty(draws) for name in declared}
    # How far Cov(E x) moves between the state the gain was taken at and the state it produced,
    # and what re-evaluating it there would have changed, both on the full declaration only.
    gain_point_shift = np.empty(draws)
    reiterated_shift = np.empty(draws)
    reiterated_nees = np.empty(draws)

    for k in range(draws):
        E = rng.multivariate_normal(np.zeros(m * n), truth).reshape(m, n)
        A = A_BAR + E                                    # the relation that actually holds
        b = A @ x_true + rng.multivariate_normal(np.zeros(m), np.diag(B_VAR))
        x_hat = x_true + rng.multivariate_normal(np.zeros(n), P)

        error = x_hat - x_true
        nees[NO_PROJECTION][k] = _nees(error, P)
        sq_error[NO_PROJECTION][k] = float(error @ error)
        reported_var[NO_PROJECTION][k] = float(np.trace(P))

        for name, A_var in declared.items():
            cs = ConstraintSet("null-draw", A_BAR, b, "the declared relation",
                               b_var=B_VAR, A_var=A_var)
            x_star, P_star = project_hard(x_hat, P, cs)
            error = x_star - x_true
            nees[name][k] = _nees(error, P_star)
            sq_error[name][k] = float(error @ error)
            reported_var[name][k] = float(np.trace(P_star))
            correction[name][k] = float(np.linalg.norm(x_star - x_hat))
            if name == FULL:
                Q_at_gain = matrix_uncertainty(cs, x_hat, P)
                Q_at_output = matrix_uncertainty(cs, x_star, P_star)
                gain_point_shift[k] = float(np.linalg.norm(Q_at_output - Q_at_gain)
                                            / np.linalg.norm(Q_at_gain))
                x_two, P_two = _reprojected(x_hat, cs, x_star, P_star)
                reiterated_shift[k] = float(np.linalg.norm(x_two - x_star)
                                            / max(np.linalg.norm(x_star - x_hat), 1e-300))
                reiterated_nees[k] = _nees(x_two - x_true, P_two)

    threshold = chi2_quantile(n, COVERAGE_LEVEL)
    return {
        "state_scale": scale,
        "state": (STATE * scale).tolist(),
        "draws": draws,
        "seed": seed,
        "dimension": n,
        "coverage_level": COVERAGE_LEVEL,
        "nees_threshold": float(threshold),
        "by_arm": {
            name: {
                "nees_mean": float(nees[name].mean()),
                "nees_over_dimension": float(nees[name].mean() / n),
                "coverage": float(np.mean(nees[name] <= threshold)),
                "rms_error": float(np.sqrt(sq_error[name].mean())),
                "reported_rms": float(np.sqrt(reported_var[name].mean())),
                "overconfidence": float(np.sqrt(sq_error[name].mean() / reported_var[name].mean())),
                "correction_norm_mean": (float(correction[name].mean())
                                         if name in correction else 0.0),
            } for name in arms},
        "one_step_gain": {
            "cov_ex_relative_shift_mean": float(gain_point_shift.mean()),
            "cov_ex_relative_shift_max": float(gain_point_shift.max()),
            "reiterated_move_over_correction_mean": float(reiterated_shift.mean()),
            "reiterated_nees_mean": float(reiterated_nees.mean()),
            "reiterated_nees_change": float(reiterated_nees.mean() - nees[FULL].mean()),
        },
    }


def quadratic_fit(experiments: list[dict]) -> dict:
    """Is the over-confidence a quadratic in the state, as the algebra says it must be?

    Cov(E x) is a quadratic form in the state and nothing else in S moves with the operating
    point, so treating A as exact should cost `1 + c * scale^2` in NEES/n -- an intercept at
    the calibrated value, and a coefficient fixed by one point. The coefficient is taken from
    the largest scale in the sweep and the prediction is then checked at every other, which is
    a real test: a mis-declaration of any constant would fit a flat line, not this.
    """
    scales = [e["state_scale"] for e in experiments]
    measured = [e["by_arm"][EXACT]["nees_over_dimension"] for e in experiments]
    c = (measured[-1] - 1.0) / scales[-1] ** 2
    predicted = [1.0 + c * s ** 2 for s in scales]
    excess = [m - 1.0 for m in measured]
    return {
        "model": "NEES/n = 1 + c * state_scale^2, c fitted at the largest scale alone",
        "coefficient": float(c),
        "state_scale": scales,
        "measured": measured,
        "predicted": predicted,
        "relative_error": [float(abs(p - m) / m) for p, m in zip(predicted, measured)],
        "worst_relative_error": float(max(abs(p - m) / m for p, m in zip(predicted, measured))),
        "excess_over_calibration": excess,
        "excess_ratio_per_doubling": [float(b / a) for a, b in zip(excess, excess[1:])],
    }


def compute() -> dict:
    experiments = [draw_projections(scale) for scale in STATE_SCALE_SWEEP]
    out = {
        "schema_version": "fsre-eiv-projection-v1",
        "scope": "A calibration measurement of the PROJECTION against a null that is true by "
                 "construction. No real record appears; nothing here describes an instrument.",
        "provenance": {"generation": provenance()},
        "declared": {
            "A_bar": A_BAR.tolist(),
            "state_at_unit_scale": STATE.tolist(),
            "P_diagonal": np.diag(P).tolist(),
            "b_var": B_VAR.tolist(),
            "shared_row_covariance": SHARED,
            "state_scale_swept": list(STATE_SCALE_SWEEP),
            "draws": DRAWS, "seed": SEED,
            "coverage_level": COVERAGE_LEVEL,
            "null_status": "true by construction in every draw: b is generated by the realised "
                           "relation and the state estimate is drawn from its declared covariance",
            "shared_with": "results/errors_in_variables -- same relation, state, covariances "
                           "and declarations, scored on the estimate instead of the statistic",
        },
        "experiments": experiments,
        "quadratic_fit": quadratic_fit(experiments),
        "limitations": [
            "The null is true by construction here, which is what makes this a calibration measurement and not a fault benchmark.",
            "Cov(E x) is evaluated at the incoming state, one step rather than iterated to a fixed point. What that costs is measured above rather than assumed small.",
            "Cov(E x) is first order in the coefficient error, and the projection inherits that approximation from the statistic unchanged.",
            "Nothing here declares dependence between A's error and b's, or between A's error and the state estimate. Shared evidence produces exactly those.",
            "One relation shape, one Gaussian coefficient error, one axis of operating point. A different relation could be more or less sensitive to each.",
            "Only the CONSTRAINT relation carries declared uncertainty. The filters' observation model H is still declared exact wherever a Kalman gain is formed against it, and an uncertain H would need the same treatment in its own place; nothing here measures or claims it.",
            "A calibrated projection is not a correct one: a well-declared A_var makes the estimate honest about what the relation can tell it, and where it can tell it little the honest answer is to leave the state near its prior.",
        ],
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition."""
    xs = r["experiments"]
    n = xs[0]["dimension"]

    def arm(e, name):
        return e["by_arm"][name]

    full_nees = [arm(e, FULL)["nees_over_dimension"] for e in xs]
    scales = [e["state_scale"] for e in xs]
    worst = max(xs, key=lambda e: e["state_scale"])
    best_known = min(xs, key=lambda e: e["state_scale"])
    return {
        "treating_an_uncertain_relation_as_exact_is_overconfident_at_every_scale": all(
            arm(e, EXACT)["nees_over_dimension"] > 1.25 for e in xs),
        "the_errors_in_variables_projection_is_calibrated_at_every_scale": all(
            abs(arm(e, FULL)["nees_over_dimension"] - 1.0) < 0.25 for e in xs),
        "the_cost_of_treating_it_as_exact_grows_with_the_operating_point": all(
            a < b for a, b in zip(r["quadratic_fit"]["measured"],
                                  r["quadratic_fit"]["measured"][1:])),
        "the_overconfidence_is_quadratic_in_the_state_with_a_calibrated_intercept":
            r["quadratic_fit"]["worst_relative_error"] < 0.05,
        "the_excess_over_calibration_quadruples_for_every_doubling_of_the_state": all(
            abs(ratio - 4.0) < 0.5
            for ratio in r["quadratic_fit"]["excess_ratio_per_doubling"]),
        "every_step_of_the_sweep_is_a_doubling": all(
            abs(b / a - 2.0) < 1e-9 for a, b in zip(scales, scales[1:])),
        "the_calibrated_projection_does_not_drift_with_the_operating_point": (
            max(full_nees) - min(full_nees) < 0.25),
        "at_the_largest_scale_projecting_as_exact_is_worse_than_not_projecting": (
            arm(worst, EXACT)["rms_error"] > arm(worst, NO_PROJECTION)["rms_error"]),
        "the_errors_in_variables_projection_is_never_worse_than_not_projecting": all(
            arm(e, FULL)["rms_error"] <= arm(e, NO_PROJECTION)["rms_error"] for e in xs),
        "where_the_relation_is_well_known_relative_to_the_state_the_projection_still_helps": (
            arm(best_known, FULL)["rms_error"] < arm(best_known, NO_PROJECTION)["rms_error"]),
        "the_declared_uncertainty_shrinks_the_correction_rather_than_only_widening_the_report": all(
            arm(e, FULL)["correction_norm_mean"] < arm(e, EXACT)["correction_norm_mean"]
            for e in xs),
        "the_shrinkage_of_the_correction_grows_with_the_operating_point": all(
            a < b for a, b in zip(
                [arm(e, EXACT)["correction_norm_mean"] / arm(e, FULL)["correction_norm_mean"]
                 for e in xs],
                [arm(e, EXACT)["correction_norm_mean"] / arm(e, FULL)["correction_norm_mean"]
                 for e in xs][1:])),
        "no_projection_is_calibrated_by_construction": all(
            abs(arm(e, NO_PROJECTION)["nees_over_dimension"] - 1.0) < 0.1 for e in xs),
        "the_one_step_gain_costs_little_at_every_scale": all(
            abs(e["one_step_gain"]["reiterated_nees_change"]) < 0.1 * n for e in xs),
        "the_sweep_spans_a_factor_of_at_least_eight_in_the_operating_point":
            scales[-1] / scales[0] >= 8.0,
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    d = report["declared"]
    xs = report["experiments"]
    n = xs[0]["dimension"]
    unit = next(e for e in xs if e["state_scale"] == 1.0)
    fit = report["quadratic_fit"]
    worst = max(xs, key=lambda e: e["state_scale"])
    best = min(xs, key=lambda e: e["state_scale"])

    A("# What treating an uncertain relation as exact costs the estimate")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A("`results/errors_in_variables` measured the first half of this: a set declaring `A_var` "
      "gets a widened `S` in the consistency statistic and in `detectability`, and without it "
      "the test rejects a system behaving exactly as declared. The projection was the half "
      "left open. `project_hard` and `project_soft` branched on `b_var` alone, so a set that "
      "declared its coefficients uncertain was reconciled against `A_bar` as though they were "
      "exact — silently, while the two functions above them in the same module refused to "
      "answer at all without a state. That was the gap the calibration study listed against "
      "itself. This closes it and prices it.")
    A("")
    A("The algebra is short. `E` is independent of the state error and zero-mean, so the "
      "cross-covariance between the state error and the residual — the numerator of the gain — "
      "does not change. Only `S` widens, by `Cov(E x)`. The gain therefore shrinks, **and it "
      "shrinks by a quadratic form in the state**. That last part is the testable one: no "
      "constant inflation of `Sigma_b` can imitate it. So the operating point is swept over a "
      f"factor of {d['state_scale_swept'][-1] / d['state_scale_swept'][0]:.0f} with every "
      "declared uncertainty held fixed.")
    A("")
    A(f"Each row is {d['draws']:,} draws in which **the declared model is true**: `b` is "
      "generated by the realised relation and the state estimate is drawn from its declared "
      f"covariance. NEES is `(x* - x_true)^T P*^-1 (x* - x_true)`; a calibrated estimate puts "
      f"it at the state dimension, {n}, so the column below reports it divided by {n} and "
      "**1.00 is calibrated**.")
    A("")

    A("## Calibration against the operating point")
    A("")
    A("| state scale | no projection | A treated as exact | `1 + c·scale²` | A_var declared |")
    A("| ---: | ---: | ---: | ---: | ---: |")
    for e, predicted in zip(xs, fit["predicted"]):
        A(f"| {e['state_scale']:g}× | "
          f"{e['by_arm'][NO_PROJECTION]['nees_over_dimension']:.2f} | "
          f"{e['by_arm'][EXACT]['nees_over_dimension']:.2f} | "
          f"{predicted:.2f} | "
          f"{e['by_arm'][FULL]['nees_over_dimension']:.2f} |")
    A("")
    A(f"Treating the relation as exact is over-confident at every scale, and the "
      f"over-confidence **grows with the operating point** — from "
      f"{best['by_arm'][EXACT]['nees_over_dimension']:.2f} at {best['state_scale']:g}× to "
      f"{worst['by_arm'][EXACT]['nees_over_dimension']:.2f} at {worst['state_scale']:g}×. The "
      f"declared version sits within "
      f"{max(abs(e['by_arm'][FULL]['nees_over_dimension'] - 1.0) for e in xs):.2f} of "
      f"calibration across the whole sweep and does not drift.")
    A("")
    A("The fourth column is the test, not a summary. `Cov(E x)` is a quadratic form in the "
      "state and nothing else in `S` moves with the operating point, so the cost of omitting "
      "it has to be `1 + c·scale²` — an intercept at the calibrated value and one free "
      f"coefficient. `c = {fit['coefficient']:.3g}` is taken from the {worst['state_scale']:g}× "
      f"row alone and predicts every other row to within "
      f"{fit['worst_relative_error'] * 100:.2f}%. Put the other way: the excess over "
      "calibration is "
      + ", ".join(f"{v:.2f}" for v in fit["excess_over_calibration"])
      + ", which **quadruples for every doubling of the state** ("
      + ", ".join(f"{v:.2f}" for v in fit["excess_ratio_per_doubling"])
      + ").")
    A("")
    A("That is the mechanism showing up where the algebra says it must, and it is the part "
      "that could have come out otherwise. A constraint set mis-declared by any fixed amount "
      "— a wrong `Sigma_b`, an optimistic `P` — would have cost the same at every scale and "
      "left this column flat. The slope is specifically the relation's own coefficients acting "
      "on a larger state, and it is the reason a declared `A_var` cannot be imitated by "
      "inflating something else.")
    A("")
    A(f"Coverage says the same thing in the units a reader acts on. At {worst['state_scale']:g}× "
      f"the nominal 95% ellipsoid actually contains the truth "
      f"{worst['by_arm'][EXACT]['coverage'] * 100:.1f}% of the time when `A` is treated as "
      f"exact, and {worst['by_arm'][FULL]['coverage'] * 100:.1f}% when it is declared.")
    A("")

    A("## Whether the estimate is better, or only more honest")
    A("")
    A("Calibration is a claim about the reported covariance. It would be a thin result if the "
      "estimate itself were no better, so the same draws are scored on RMS error against the "
      "truth, with not projecting at all as the baseline.")
    A("")
    A("| state scale | no projection | A treated as exact | A_var declared |")
    A("| ---: | ---: | ---: | ---: |")
    for e in xs:
        A(f"| {e['state_scale']:g}× | "
          f"{e['by_arm'][NO_PROJECTION]['rms_error']:.4f} | "
          f"{e['by_arm'][EXACT]['rms_error']:.4f} | "
          f"{e['by_arm'][FULL]['rms_error']:.4f} |")
    A("")
    A(f"At {best['state_scale']:g}× the coefficients are well known relative to the state and "
      f"the projection earns its place: RMS error falls from "
      f"{best['by_arm'][NO_PROJECTION]['rms_error']:.4f} to "
      f"{best['by_arm'][FULL]['rms_error']:.4f}. At {worst['state_scale']:g}× the same "
      f"constraint has little left to say, and **projecting as though `A` were exact is worse "
      f"than not projecting at all** — {worst['by_arm'][EXACT]['rms_error']:.4f} against "
      f"{worst['by_arm'][NO_PROJECTION]['rms_error']:.4f}. The declared version never is, at "
      f"any scale in the sweep.")
    A("")
    A("That is the behaviour worth naming. The correction shrinks rather than the report merely "
      "widening: mean correction norm "
      f"{unit['by_arm'][EXACT]['correction_norm_mean']:.4f} treated as exact against "
      f"{unit['by_arm'][FULL]['correction_norm_mean']:.4f} declared at 1×, and the ratio grows "
      "with the operating point. Where an uncertain relation can no longer say anything useful "
      "about the state, the declared projection degrades towards leaving the state where it "
      "found it, instead of confidently moving it somewhere wrong.")
    A("")

    A("## What the one-step gain costs")
    A("")
    A("`Cov(E x)` is a quadratic form in the state, so unlike every other case the gain depends "
      "on where it is evaluated, and the kernel evaluates it at the incoming state — one step, "
      "not iterated to a fixed point. There is no honest way to argue that away, so it is "
      "measured: the table re-evaluates `Cov(E x)` at the update's own output and redoes the "
      "same update from the same prior.")
    A("")
    A("| state scale | Cov(E x) shift at the output | re-iterated move / correction | NEES change |")
    A("| ---: | ---: | ---: | ---: |")
    for e in xs:
        g = e["one_step_gain"]
        A(f"| {e['state_scale']:g}× | {g['cov_ex_relative_shift_mean'] * 100:.2f}% | "
          f"{g['reiterated_move_over_correction_mean'] * 100:.2f}% | "
          f"{g['reiterated_nees_change']:+.4f} |")
    A("")
    A(f"The largest NEES change iterating would buy anywhere in the sweep is "
      f"{max(abs(e['one_step_gain']['reiterated_nees_change']) for e in xs):.4f} against a "
      f"target of {n}. The one-step gain stays, and it stays because this says it may, not "
      f"because iterating would have been inconvenient.")
    A("")
    A(f"Worth reading the first column the other way round: `Cov(E x)` moves most between the "
      f"two states at {best['state_scale']:g}× "
      f"({best['one_step_gain']['cov_ex_relative_shift_mean'] * 100:.1f}%), which is the end "
      f"of the sweep where the relation is best known and therefore corrects the state most. "
      f"Where the gain is large the evaluation point moves; where the evaluation point matters "
      f"most to the answer the gain is small. The two effects sit at opposite ends, which is "
      f"why one step is enough across the whole range rather than only in the easy part of it.")
    A("")

    A("## What this does not fix")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "eiv_projection.md").write_text(text, encoding="utf-8")
    (out_dir / "eiv_projection.json").write_text(
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
