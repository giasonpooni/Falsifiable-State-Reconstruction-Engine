"""Estimators. All expose the same two-call interface:

    ingest(obs, j)   -- absorb the observation sampled at step j; the runner calls this
                        only once the observation has *arrived* (obs.arrival_t <= now),
                        always in sampling order
    report(k)        -- the state at step k, predicted forward from the last ingested
                        sampling step with the commanded inputs

Delay therefore enters only through Observation.arrival_t. Arrival order never
substitutes for sampling time, and nothing is read before it has arrived.

Every estimator is initialised from a DECLARED prior (mean and std), never
from the simulator's state.

Every estimator also keeps an innovation record, one row per ingested
sampling step j and one column per sensor:

    innov[j]      nu   = y - H x_pred          (NaN where the sensor is missing)
    innov_var[j]  S_ii = (H P_pred H^T + R)_ii
    innov_z[j]    z    = nu / sqrt(S_ii)

The runner copies these into RunResult and runs the per-sensor CUSUM on z.
Hold-last has no prediction and records NaN.

The first two components of every reported state are the reservoir masses
(m1, m2); the reconciliation stage acts on that marginal only. An estimator
that carries more state declares the extra components in `aug_names` (with
the no-fault value of each in `aug_nominal`); the runner stores those in
RunResult.extra and flags each one when it departs from its nominal value by
more than its own reported uncertainty explains.
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
    config_cls = KFConfig
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float, ...] = ()

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
    config_cls = KFConfig
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float, ...] = ()

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


@dataclass(frozen=True)
class AugConfig:
    """Configuration of the augmented-state filter. Values are the ones stated in the
    stage goal; none of them is tuned to a scenario."""
    sigma_w: float = 0.05     # per-step process std per reservoir, kg (as KFConfig; diagonal Q)
    q_alpha: float = 1e-3     # per-step random-walk std of the pump scale alpha (dimensionless)
    q_L: float = 2e-3         # per-step random-walk std of the boundary flux L, kg/s
    alpha0: float = 1.0       # declared prior mean of alpha: the pump delivers what is commanded
    alpha0_std: float = 0.1
    L0: float = 0.0           # declared prior mean of L: the boundary is closed
    L0_std: float = 0.02


class AugmentedKalmanFilter:
    """Time-varying linear KF for x = [m1, m2, alpha, L]: the pump scale and the boundary
    flux are states with their own uncertainty instead of assumptions.

        m1'    = m1 - alpha u dt
        m2'    = m2 + alpha u dt - L dt
        alpha' = alpha
        L'     = L

    With u the COMMANDED pump rate at step k, this is linear in the state:

        F_k = [[1, 0, -u dt,  0 ],
               [0, 1,  u dt, -dt],
               [0, 0,  1,     0 ],
               [0, 0,  0,     1 ]]

    so it is run as a linear KF with a time-varying F, not as an EKF. Q is diagonal:
    sigma_w^2 per reservoir (as KFConfig, so closure is still not the filter's claim),
    q_alpha^2 and q_L^2 for the two parameters, which lets them drift. H observes m1
    and m2 only (masked as for KalmanFilter); alpha is observable only while u != 0
    and L only through m2, so each parameter's reported variance grows by its own
    random walk whenever the evidence cannot see it.

    AugConfig defaults: sigma_w = 0.05 kg, q_alpha = 1e-3 per step, q_L = 2e-3 kg/s per
    step, prior alpha ~ N(1.0, 0.1^2), prior L ~ N(0.0, 0.02^2). The declared prior for
    the mass part comes from run() as for the other estimators.

    report(k) returns the full 4-state (x (4,), P (4, 4)). The runner keeps the mass
    marginal (x[:2], P[:2, :2]) in RunResult.x / P and puts alpha_hat, alpha_sd,
    L_hat, L_sd and the two debounced flags in RunResult.extra.
    """

    model_version = "reservoir2-aug-alpha-L-v1"
    config_cls = AugConfig
    aug_names = ("alpha", "L")
    aug_nominal = (1.0, 0.0)   # no-fault values the runner's flags test against (NOT the prior)
    N = 4

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: AugConfig = AugConfig()):
        self.dt = dt
        self.u = u_cmd
        self.cfg = cfg
        self.Q = np.diag([cfg.sigma_w ** 2, cfg.sigma_w ** 2, cfg.q_alpha ** 2, cfg.q_L ** 2])
        m0 = np.asarray(x0, dtype=float)
        self.x0 = np.array([m0[0], m0[1], cfg.alpha0, cfg.L0])
        self.P0 = np.diag([float(p0_std) ** 2, float(p0_std) ** 2, cfg.alpha0_std ** 2, cfg.L0_std ** 2])
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    def F(self, k: int) -> np.ndarray:
        q = self.u[k] * self.dt
        return np.array([[1.0, 0.0, -q, 0.0],
                         [0.0, 1.0, q, -self.dt],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def predict(self, x, P, k):
        F = self.F(k)
        return F @ x, F @ P @ F.T + self.Q

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.xf):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, expected {len(self.xf)}")
        if j > 0:
            self._x, self._P = self.predict(self._x, self._P, j - 1)
        m = obs.mask
        nu_full = np.full(2, np.nan)
        s_full = np.full(2, np.nan)
        if m.any():
            H = np.eye(self.N)[:2][m]                          # sensors read m1, m2 only
            y = obs.y[m]
            R = obs.R[np.ix_(m, m)]
            S = H @ self._P @ H.T + R
            K = self._P @ H.T @ np.linalg.inv(S)
            nu = y - H @ self._x
            self._x = self._x + K @ nu
            I_KH = np.eye(self.N) - K @ H
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


ESTIMATORS = {"kf": KalmanFilter, "hold_last": HoldLast, "kf_aug": AugmentedKalmanFilter}
