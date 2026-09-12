"""A conserving record, an identifiable step, and a confounded drift/flow offset."""
from dataclasses import replace

import numpy as np

from set_lcm.diagnostics import diagnose
from set_lcm.measurement import BalanceRecord, balance_residuals


def main():
    edges = np.arange(7, dtype=float)  # six one-second intervals
    storage = 100.0 + edges
    flows = np.tile([3.0, 2.0], (6, 1))  # inflow minus outflow = 1 m3/s
    record = BalanceRecord(
        edges=edges, storage=storage, flows=flows, flow_signs=np.array([1.0, -1.0]),
        covariance=np.r_[np.full(7, 0.05**2), np.full(12, 0.02**2)],
        storage_support="instant", flow_support="mean", volume_unit="m3", flow_unit="m3/s",
    )
    H = balance_residuals(record).operator
    step = np.r_[(edges >= 3).astype(float), np.zeros(12)]
    drift = np.r_[edges, np.zeros(12)]
    flow_offset = np.r_[np.zeros(7), np.tile([1.0, 0.0], 6)]
    hypotheses = {"storage_step": H @ step, "storage_drift": H @ drift,
                  "inflow_offset": H @ flow_offset}
    for label, bias in (
        ("Conserving measurements", np.zeros(7)),
        ("Storage step of 1 m3 after 3 seconds", step[:7]),
        ("Storage drift of 0.2 m3/s", 0.2 * edges),
    ):
        series = balance_residuals(replace(record, storage=storage + bias))
        result = diagnose(series.residual, series.covariance, hypotheses)
        print(f"{label}: {result.status}")
        if result.status in ("identified", "ambiguous"):
            print("  Compatible candidates: " + ", ".join(result.candidates))
    print("\nCandidates assume known profiles and onset. Consistent does not mean healthy.")
    print("Storage drift and an inflow offset can produce the same balance residual.")


if __name__ == "__main__":
    main()
