"""Estimators. Both expose the same two-call interface:

    ingest(obs, j)   -- absorb the observation sampled at step j; the runner calls this
                        only once the observation has *arrived* (obs.arrival_t <= now),
                        always in sampling order
    report(k)        -- the state at step k, predicted forward from the last ingested
                        sampling step with the commanded inputs

Delay therefore enters only through Observation.arrival_t. Arrival order never
substitutes for sampling time, and nothing is read before it has arrived.

Both estimators are initialised from a DECLARED prior (mean and std), never
from the simulator's state.

Every estimator also keeps an innovation record, one row per ingested
sampling step j and one column per sensor:

    innov[j]      nu   = y - H x_pred          (NaN where the sensor is missing)
    innov_var[j]  S_ii = (H P_pred H^T + R)_ii
    innov_z[j]    z    = nu / sqrt(S_ii)

The runner copies these into RunResult and runs the per-sensor CUSUM on z.
Hold-last has no prediction and records NaN.
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


class KalmanFilter:
    """Linear KF for x = [m1, m2]:  x_{k+1} = x_k + B u_k dt + w,  y = H_mask x + v."""

    model_version = MODEL_VERSION

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: KFConfig = KFConfig()):
        self.dt = dt
        self.u = u_cmd
        self.cfg = cfg
        self.Q = np.eye(2) * cfg.sigma_w ** 2
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = np.eye(2) * float(p0_std) ** 2
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    def predict(self, x, P, k):
        return x + B * self.u[k] * self.dt, P + self.Q

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.xf):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, expected {len(self.xf)}")
        if j > 0:
            self._x, self._P = self.predict(self._x, self._P, j - 1)
        m = obs.mask
        nu_full = np.full(2, np.nan)
        s_full = np.full(2, np.nan)
        if m.any():
            H = np.eye(2)[m]
            y = obs.y[m]
            R = obs.R[np.ix_(m, m)]
            S = H @ self._P @ H.T + R
            K = self._P @ H.T @ np.linalg.inv(S)
            nu = y - H @ self._x                               # innovation against the prediction
            self._x = self._x + K @ nu
            I_KH = np.eye(2) - K @ H
            self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T   # Joseph form
            nu_full[m] = nu
            s_full[m] = np.diag(S)
        self.xf.append(self._x.copy())
        self.Pf.append(self._P.copy())
        self.innov.append(nu_full)
        self.innov_var.append(s_full)
        self.innov_z.append(nu_full / np.sqrt(s_full))

    def report(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        j = len(self.xf) - 1
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

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: KFConfig = KFConfig()):
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.var0 = np.full(2, float(p0_std) ** 2)
        self.inflate = cfg.sigma_w ** 2
        self._x = self.x0.copy()
        self._var = self.var0.copy()
        self.hist: list[tuple[np.ndarray, np.ndarray]] = []
        # no prediction, hence no innovation: the record is NaN throughout
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.hist):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, expected {len(self.hist)}")
        self._var = self._var + self.inflate
        m = obs.mask
        self._x[m] = obs.y[m]
        self._var[m] = np.diag(obs.R)[m]
        self.hist.append((self._x.copy(), self._var.copy()))
        nan2 = np.full(2, np.nan)
        self.innov.append(nan2.copy())
        self.innov_var.append(nan2.copy())
        self.innov_z.append(nan2.copy())

    def report(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        j = len(self.hist) - 1
        if j < 0:
            return self.x0.copy(), np.diag(self.var0 + k * self.inflate)
        x, var = self.hist[j]
        return x.copy(), np.diag(var + (k - j) * self.inflate)


ESTIMATORS = {"kf": KalmanFilter, "hold_last": HoldLast}
