"""Estimators. Both expose the same two-call interface:

    ingest(obs, k)        -- absorb the observation sampled at step k
    report(k, delay)      -- the state reportable at t_k using only observations
                             that have *arrived* by t_k

Delay is treated honestly: with a uniform arrival delay d, the reportable state
at t_k is the filtered state at k-d predicted forward d steps. Arrival order
never substitutes for sampling time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Observation
from .simulator import MODEL_VERSION

B = np.array([-1.0, 1.0])   # commanded transfer moves mass from reservoir 1 to reservoir 2


@dataclass(frozen=True)
class KFConfig:
    # Per-step process std per reservoir. Diagonal Q on purpose: the estimator
    # does NOT assume the boundary is closed -- closure is the constraint
    # module's declared claim, not the filter's.
    sigma_w: float = 0.05
    p0_std: float = 5.0


class KalmanFilter:
    """Linear KF for x = [m1, m2]:  x_{k+1} = x_k + B u_k dt + w,  y = H_mask x + v."""

    model_version = MODEL_VERSION

    def __init__(self, x0, dt: float, u_cmd: np.ndarray, cfg: KFConfig = KFConfig()):
        self.dt = dt
        self.u = u_cmd
        self.cfg = cfg
        self.Q = np.eye(2) * cfg.sigma_w ** 2
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = np.eye(2) * cfg.p0_std ** 2
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []

    def predict(self, x, P, k):
        return x + B * self.u[k] * self.dt, P + self.Q

    def ingest(self, obs: Observation, k: int) -> None:
        if k > 0:
            self._x, self._P = self.predict(self._x, self._P, k - 1)
        m = obs.mask
        if m.any():
            H = np.eye(2)[m]
            y = obs.y[m]
            R = obs.R[np.ix_(m, m)]
            S = H @ self._P @ H.T + R
            K = self._P @ H.T @ np.linalg.inv(S)
            self._x = self._x + K @ (y - H @ self._x)
            I_KH = np.eye(2) - K @ H
            self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T   # Joseph form
        self.xf.append(self._x.copy())
        self.Pf.append(self._P.copy())

    def report(self, k: int, delay: int) -> tuple[np.ndarray, np.ndarray]:
        j = k - delay
        if j < 0:
            x, P, start = self.x0.copy(), self.P0.copy(), 0
        else:
            x, P, start = self.xf[j].copy(), self.Pf[j].copy(), j
        for i in range(start, k):
            x, P = self.predict(x, P, i)
        return x, P


class HoldLast:
    """Baseline: hold the last value each sensor reported. Ignores the dynamics.
    Variance grows by sigma_w^2 per blind step so its intervals are at least honest
    about staleness."""

    model_version = "hold-last-value-v0"

    def __init__(self, x0, dt: float, u_cmd: np.ndarray, cfg: KFConfig = KFConfig()):
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.var0 = np.full(2, cfg.p0_std ** 2)
        self.inflate = cfg.sigma_w ** 2
        self._x = self.x0.copy()
        self._var = self.var0.copy()
        self.hist: list[tuple[np.ndarray, np.ndarray]] = []

    def ingest(self, obs: Observation, k: int) -> None:
        self._var = self._var + self.inflate
        m = obs.mask
        self._x[m] = obs.y[m]
        self._var[m] = np.diag(obs.R)[m]
        self.hist.append((self._x.copy(), self._var.copy()))

    def report(self, k: int, delay: int) -> tuple[np.ndarray, np.ndarray]:
        j = k - delay
        if j < 0:
            return self.x0.copy(), np.diag(self.var0 + k * self.inflate)
        x, var = self.hist[j]
        return x.copy(), np.diag(var + delay * self.inflate)


ESTIMATORS = {"kf": KalmanFilter, "hold_last": HoldLast}
