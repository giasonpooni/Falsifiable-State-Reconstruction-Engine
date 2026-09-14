"""Affine-coordinate equivalence of an additive invariant filter and ordinary KF.

This is a numerical contract experiment, not an estimation-performance benchmark.
The same records and full covariance model are presented in different coordinates.
An independently written Joseph-form KF supplies the original-coordinate reference.
Simulation truth is returned separately and is never read by either filter or scorer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from set_lcm.coordinates import AffineCoordinates
from set_lcm.invariant import GaussianState, predict, update

from .provenance import header_line, provenance

N_STEPS = 32
SEEDS = tuple(range(8))
SCENARIOS = ("clean", "biased_gauge")
ABSOLUTE_TOLERANCE = 1e-8
BIAS_ONSET = 16
GAUGE_BIAS_KG = 2.0
METRICS = ("max_abs_mean_difference_kg", "max_abs_covariance_difference_kg2", "max_abs_nis_difference")


@dataclass(frozen=True)
class MeasurementRecord:
    measurements: np.ndarray
    mask: np.ndarray


def model() -> dict[str, np.ndarray]:
    """A fixed two-tank model in kg, with conserving exchange and pump transfer."""
    return {
        "initial_mean": np.array([60.0, 40.0]),
        "initial_covariance": np.array([[4.0, 1.2], [1.2, 3.0]]),
        "F": np.array([[0.96, 0.03], [0.04, 0.97]]),
        "drift": np.array([0.2, -0.2]),
        "Q": 0.0025 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
        "H": np.eye(2),
        "observation_offset": np.array([0.7, -0.4]),
        "R": np.array([[0.16, 0.06], [0.06, 0.25]]),
    }


def measurement_mask() -> np.ndarray:
    mask = np.ones((N_STEPS, 2), dtype=bool)
    mask[2::7, 0] = False
    mask[1::5, 1] = False
    mask[[10, 24], :] = False
    return mask


def simulate(seed: int, scenario: str) -> tuple[MeasurementRecord, np.ndarray]:
    """Return only measurements to inference; keep hidden physical states separate."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    m = model()
    rng = np.random.default_rng(seed)
    x = m["initial_mean"] + np.linalg.cholesky(m["initial_covariance"]) @ rng.standard_normal(2)
    sensor_chol = np.linalg.cholesky(m["R"])
    truth, measurements = [], []
    for k in range(N_STEPS):
        exchange = 0.05 * rng.standard_normal() * np.array([1.0, -1.0])
        x = m["F"] @ x + m["drift"] + exchange
        z = m["H"] @ x + m["observation_offset"] + sensor_chol @ rng.standard_normal(2)
        if scenario == "biased_gauge" and k >= BIAS_ONSET:
            z[0] += GAUGE_BIAS_KG
        truth.append(x.copy())
        measurements.append(z)
    measurements = np.asarray(measurements)
    mask = measurement_mask()
    measurements[~mask] = np.nan
    return MeasurementRecord(measurements, mask), np.asarray(truth)


def ordinary_reference(record: MeasurementRecord) -> dict:
    """Independent affine KF: direct solve for K and Joseph covariance update.

    This function does not call the invariant layer, chart helper or simulator,
    and receives no physical states or intervention labels.
    """
    m = model()
    mean, covariance = m["initial_mean"].copy(), m["initial_covariance"].copy()
    means, covariances, statistics, statuses, dofs = [], [], [], [], []
    for z, observed in zip(record.measurements, record.mask, strict=True):
        mean = m["F"] @ mean + m["drift"]
        covariance = m["F"] @ covariance @ m["F"].T + m["Q"]
        dof = int(observed.sum())
        if dof:
            H = m["H"][observed]
            R = m["R"][np.ix_(observed, observed)]
            innovation = z[observed] - H @ mean - m["observation_offset"][observed]
            S = H @ covariance @ H.T + R
            gain = np.linalg.solve(S, H @ covariance).T
            mean = mean + gain @ innovation
            J = np.eye(2) - gain @ H
            covariance = J @ covariance @ J.T + gain @ R @ gain.T
            statistic = float(innovation @ np.linalg.solve(S, innovation))
        else:
            statistic = None
        covariance = (covariance + covariance.T) / 2
        means.append(mean.copy())
        covariances.append(covariance.copy())
        statistics.append(statistic)
        statuses.append("updated" if dof else "no_observations")
        dofs.append(dof)
    return {"means": np.asarray(means), "covariances": np.asarray(covariances),
            "statistics": statistics, "statuses": statuses, "dofs": dofs}


def state_charts() -> dict[str, AffineCoordinates]:
    return {
        "identity": AffineCoordinates(np.eye(2), np.zeros(2)),
        "tank_permutation": AffineCoordinates(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros(2)),
        "total_difference": AffineCoordinates(np.array([[1.0, 1.0], [1.0, -1.0]]), np.zeros(2)),
        "unit_scale": AffineCoordinates(np.diag([1000.0, 0.001]), np.zeros(2)),
        "datum_translation": AffineCoordinates(np.eye(2), np.array([12.0, -7.0])),
    }


def measurement_charts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Monomial maps only: new row i = scale[i] * old row order[i].

    Gathering rows before scaling avoids 0*NaN contamination from masked rows.
    Arbitrary measurement mixing cannot reuse a per-channel boolean mask.
    """
    return {
        "identity": (np.array([0, 1]), np.ones(2)),
        "reordered": (np.array([1, 0]), np.ones(2)),
        "unit_scale": (np.array([0, 1]), np.array([1000.0, 0.001])),
        "reordered_scaled": (np.array([1, 0]), np.array([0.001, 1000.0])),
    }


def transformed_run(record: MeasurementRecord, chart: AffineCoordinates,
                    row_order: np.ndarray, row_scale: np.ndarray) -> dict:
    m = model()
    state = chart.transform_state(GaussianState(m["initial_mean"], m["initial_covariance"]))
    F, drift, Q = chart.transform_dynamics(m["F"], m["drift"], m["Q"])
    H, offset = chart.transform_observation(m["H"], m["observation_offset"])
    H, offset = row_scale[:, None] * H[row_order], row_scale * offset[row_order]
    R = row_scale[:, None] * m["R"][np.ix_(row_order, row_order)] * row_scale[None, :]
    measurements = record.measurements[:, row_order] * row_scale
    masks = record.mask[:, row_order]
    means, covariances, statistics, statuses, dofs = [], [], [], [], []
    for z, observed in zip(measurements, masks, strict=True):
        prior = predict(state, F, drift, Q)
        result = update(prior, z, H, R, offset=offset, mask=observed)
        state = result.posterior
        restored = chart.restore_state(state)
        means.append(restored.mean)
        covariances.append(restored.covariance)
        statistics.append(result.statistic)
        statuses.append(result.status)
        dofs.append(result.dof)
    return {"means": np.asarray(means), "covariances": np.asarray(covariances),
            "statistics": statistics, "statuses": statuses, "dofs": dofs}


def compare_traces(reference: dict, transformed: dict) -> dict:
    statistic_pairs = [(a, b) for a, b in zip(reference["statistics"], transformed["statistics"], strict=True)
                       if a is not None and b is not None]
    metrics = {
        "max_abs_mean_difference_kg": float(np.max(np.abs(reference["means"] - transformed["means"]))),
        "max_abs_covariance_difference_kg2": float(np.max(np.abs(reference["covariances"] - transformed["covariances"]))),
        "max_abs_nis_difference": max((abs(a - b) for a, b in statistic_pairs), default=0.0),
        "statuses_match": reference["statuses"] == transformed["statuses"],
        "dofs_match": reference["dofs"] == transformed["dofs"],
        "statistic_presence_matches": [v is None for v in reference["statistics"]] ==
                                      [v is None for v in transformed["statistics"]],
        "statuses": transformed["statuses"], "dofs": transformed["dofs"],
    }
    metrics["equivalent"] = (all(np.isfinite(metrics[key]) and metrics[key] <= ABSOLUTE_TOLERANCE for key in METRICS)
                             and metrics["statuses_match"] and metrics["dofs_match"]
                             and metrics["statistic_presence_matches"])
    return metrics


def run_experiment() -> dict:
    charts, observation_charts = state_charts(), measurement_charts()
    records = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            record, _ = simulate(seed, scenario)
            reference = ordinary_reference(record)
            runs = {}
            for chart_name, chart in charts.items():
                for observation_name, (order, scale) in observation_charts.items():
                    transformed = transformed_run(record, chart, order, scale)
                    runs[f"{chart_name}/{observation_name}"] = compare_traces(reference, transformed)
            records.append({"scenario": scenario, "seed": seed,
                            "reference_statuses": reference["statuses"], "reference_dofs": reference["dofs"],
                            "runs": runs})
    outcomes = [run for record in records for run in record["runs"].values()]
    return {
        "schema_version": "fsre-invariant-layer-v1", "provenance": provenance(),
        "design": {
            "scope": "Numerical equivalence and coordinate contracts for the additive Gaussian layer; not a performance or fault-classification benchmark.",
            "n_steps": N_STEPS, "seeds": list(SEEDS), "scenarios": list(SCENARIOS),
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "tolerance_units": "Original kg for means, kg^2 for covariance entries, dimensionless for NIS; fixed before running.",
            "noise": "Independent process and observation noise across times and from the initial error; full correlated R retained. Q is conserving random exchange.",
            "missingness": measurement_mask().tolist(),
            "biased_gauge": {"index": 0, "offset_kg": GAUGE_BIAS_KG, "onset_step": BIAS_ONSET},
            "reference": "Independent direct-solve ordinary affine KF with Joseph covariance; receives observations and masks, never physical truth or fault labels.",
            "pairing": "The same eight noise seeds are reused across scenarios and charts. These are paired numerical comparisons, not independent performance trials.",
            "order": "Predict, then assimilate each sample jointly using the observed principal R submatrix. Sensor permutation/scaling preserves its mask and correlations.",
            "chart_interpretation": "Fixed affine coordinate changes x'=T x+c, not physical interventions. Translation changes the coordinate origin, not an automorphism of the zero-identity additive group; centered errors transform as T e.",
            "statistic": "NIS is the pre-update innovation Mahalanobis statistic. Update/no_observations statuses are processing outcomes, not fault verdicts. No alarm threshold is used.",
            "limits": "The additive layer equals ordinary KF for this affine Gaussian model. No nonlinear IEKF accuracy, robustness, calibration or sensor-identification gain is claimed. Coordinate changes cannot resolve existing fault ambiguity.",
        },
        "model": {key: value.tolist() for key, value in model().items()},
        "state_charts": {name: {"matrix": chart.matrix.tolist(), "offset": chart.offset.tolist()}
                         for name, chart in charts.items()},
        "measurement_charts": {name: {"row_order": order.tolist(), "row_scale": scale.tolist()}
                               for name, (order, scale) in observation_charts.items()},
        "summary": {**{key: max(row[key] for row in outcomes) for key in METRICS},
                    "compared_runs": len(outcomes), "equivalent_runs": sum(row["equivalent"] for row in outcomes),
                    "all_equivalent": all(row["equivalent"] for row in outcomes)},
        "records": records,
    }


def render(report: dict) -> str:
    lines = ["# Additive invariant-layer equivalence", "", header_line(report["provenance"]), "",
             "**This tests numerical equivalence to ordinary Kalman filtering, not a performance improvement.**", "",
             "Five fixed state charts and four measurement order/unit charts use the same two-tank records. "
             "All means and covariances are mapped back before comparison. Processing statuses must match exactly.", "",
             f"Fixed absolute tolerance: {report['design']['absolute_tolerance']:g} in original kg, kg² and dimensionless NIS units.", "",
             "| scenario | seed | largest mean difference [kg] | largest covariance difference [kg²] | largest NIS difference | equivalent chart runs |",
             "|---|---:|---:|---:|---:|---:|"]
    for record in report["records"]:
        rows = list(record["runs"].values())
        lines.append(f"| {record['scenario']} | {record['seed']} | "
                     + " | ".join(f"{max(row[key] for row in rows):.3e}" for key in METRICS)
                     + f" | {sum(row['equivalent'] for row in rows)}/{len(rows)} |")
    summary = report["summary"]
    lines += ["", f"Equivalent comparisons: {summary['equivalent_runs']}/{summary['compared_runs']}.", "",
              "The JSON retains the model, charts, masks and per-seed/per-chart discrepancies, degrees of freedom and statuses.", "",
              "## Interpretation", ""]
    for key in ("noise", "reference", "pairing", "order", "chart_interpretation", "statistic", "limits"):
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {report['design'][key]}")
    lines += ["", "The biased-gauge scenario deliberately violates the no-bias observation model. "
              "Equivalence there means both implementations apply the same model to the same faulty record; "
              "it is not evidence that either recovers the true state or identifies the faulty gauge.", "",
              "Measurement scaling would change Gaussian log density by its Jacobian term; this report "
              "compares the invariant NIS instead of claiming raw log-likelihood equality.", ""]
    return "\n".join(lines)


def main(out_dir: Path, *, quiet: bool = False) -> int:
    report = run_experiment()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "invariant_layer.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render(report)
    (out_dir / "invariant_layer.md").write_text(text, encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
