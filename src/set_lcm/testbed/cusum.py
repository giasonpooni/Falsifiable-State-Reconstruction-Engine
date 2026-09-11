"""Per-sensor two-sided CUSUM on the normalised innovation.

This is the evidence side's own fault-detection channel. It reads nothing but
the filter's normalised innovation z = nu / sqrt(S_ii) for each sensor, so it
is independent of the declared constraint and of the reconciliation stage:
it can see faults that lie in null(A), and it cannot see anything that does
not move a sensor's innovation.

    g+ <- max(0, g+ + z - k)        g- <- max(0, g- - z - k)
    alarm when max(g+, g-) > h, then reset that channel to zero

k is the reference value (half the shift, in units of the innovation's own
standard deviation, that the test is tuned to catch fastest); h is the
decision interval. A NaN input (sensor missing at that step) leaves the
channel untouched. results/calibration.md measures the null of this statistic
on the shipped scenarios for h in {4, 6, 8, 10}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CusumConfig:
    k: float = 0.5   # reference value (allowance) in units of sqrt(S_ii)
    h: float = 8.0   # decision interval; float("inf") disables alarms and resets


class Cusum:
    """Incremental two-sided CUSUM over `n` parallel channels.

    `stat[i]` is max(g+, g-) after the most recent update of channel i, *before*
    the reset an alarm applies, so at an alarm step it shows the value that
    crossed h.
    """

    def __init__(self, cfg: CusumConfig, n: int = 2):
        self.cfg = cfg
        self.g_plus = np.zeros(n)
        self.g_minus = np.zeros(n)
        self.stat = np.zeros(n)

    def update(self, z: np.ndarray) -> np.ndarray:
        """Feed one normalised-innovation vector; return a bool alarm vector."""
        z = np.asarray(z, dtype=float)
        alarm = np.zeros(z.shape[0], dtype=bool)
        k, h = self.cfg.k, self.cfg.h
        for i, zi in enumerate(z):
            if np.isnan(zi):
                continue
            self.g_plus[i] = max(0.0, self.g_plus[i] + zi - k)
            self.g_minus[i] = max(0.0, self.g_minus[i] - zi - k)
            self.stat[i] = max(self.g_plus[i], self.g_minus[i])
            if self.stat[i] > h:
                alarm[i] = True
                self.g_plus[i] = 0.0
                self.g_minus[i] = 0.0
        return alarm


def cusum_sequence(z: np.ndarray, cfg: CusumConfig = CusumConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Run the recursion over a 1-D sequence. Returns (stat, alarm), each of length len(z);
    stat[t] is the value after step t before any reset, alarm[t] marks the crossings."""
    z = np.asarray(z, dtype=float).reshape(-1)
    c = Cusum(cfg, n=1)
    stat = np.zeros(z.size)
    alarm = np.zeros(z.size, dtype=bool)
    for t, zt in enumerate(z):
        alarm[t] = c.update(np.array([zt]))[0]
        stat[t] = c.stat[0]
    return stat, alarm
