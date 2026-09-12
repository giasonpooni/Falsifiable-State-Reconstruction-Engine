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

Every estimator declares `n_report`, the number of leading components of its
reported state that the reconciliation stage acts on (2 for every estimator
in this module: the reservoir masses m1, m2; the water-level filters that
estimators_water adds to ESTIMATORS report 1, the level). An estimator that
carries more state declares the extra components, which follow those
n_report, in `aug_names`, with the no-fault value of each in `aug_nominal`;
the runner stores them in RunResult.extra and flags each one when it departs
from its nominal value by more than its own reported uncertainty explains. A nominal of None declares
"no flag for this component": its estimate and sd are recorded, nothing tests
them. The number of sensors is the observation's (obs.y.size); every
innovation-record row has one entry per sensor.

Feedback. An estimator that can accept a reconciled state back exposes

    set_state(x, P)  -- replace the state at the last ingested sampling step

which the runner calls, for EstimatorSpec.feedback = True, with the projected
(x*, P*) of the mass marginal after a projection was applied. P* is
rank-deficient along the constraint by construction; the next predict adds Q,
so every REPORTED covariance is SPD again. KalmanFilter and its structural-Q
variant implement it; HoldLast (no covariance to speak of) and the augmented
filter (the mass marginal cannot be replaced without its cross-covariances
with alpha and L) raise NotImplementedError.

Baselines that could remove the case for a constraint row:

    kf_closedq   the same KF with closure written into Q instead of into a
                 constraint: Q = sigma_q^2 dt^2 B B^T + eps I. It "knows" the
                 boundary is closed through its noise model.
    oracle       a KF given the HIDDEN actual pump rate and leak as known
                 inputs, Q = eps I only. A BOUND on what a perfect model of the
                 inputs could do, never a candidate. It receives the hidden arrays
                 only through the explicit `oracle_inputs` argument, which
                 runner.run() forwards for kind "oracle" and refuses for every
                 other kind; experiment code is what supplies them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Observation
from .estimators_balance import WaterBalanceAugmented, WaterBalanceClosed, WaterBalanceOpen
from .estimators_water import LevelTrendKF, TideKF
from .simulator import MODEL_VERSION

B = np.array([-1.0, 1.0])   # commanded transfer moves mass from reservoir 1 to reservoir 2
LEAK_DIR = np.array([0.0, -1.0])   # a leak drains reservoir 2 (simulator: dm2/dt = q - leak)


@dataclass(frozen=True)
class KFConfig:
    # Per-step process std per reservoir. Diagonal Q on purpose: the estimator
    # does NOT assume the boundary is closed -- closure is the constraint
    # module's declared claim, not the filter's.
    sigma_w: float = 0.05


@dataclass(frozen=True)
class ClosedQConfig:
    """Structurally closed process noise: Q = sigma_q^2 dt^2 B B^T + eps I.

    sigma_q is the simulator's declared per-step pump fluctuation
    (SimConfig.pump_noise_std = 0.01 kg/s), i.e. a stated model parameter, not a
    value tuned to a scenario. eps keeps Q (and so P) positive definite so the
    kernel's guards accept the reported covariance."""
    sigma_q: float = 0.01   # kg/s, along B = (-1, +1): mass moves between the reservoirs, the sum does not
    eps: float = 1e-8       # kg^2 per step, isotropic floor


@dataclass(frozen=True)
class OracleConfig:
    """The oracle's only free parameter: the isotropic process-noise floor."""
    eps: float = 1e-8       # kg^2 per step


class KalmanFilter:
    """Linear KF for x = [m1, m2]:  x_{k+1} = x_k + B u_k dt + w,  y = H_mask x + v.

    Subclasses change the process noise (`process_noise`) and the deterministic
    per-step drift (`drift`); everything else -- prior, H, Joseph update, the
    innovation record, reporting and feedback -- is shared."""

    model_version = MODEL_VERSION
    config_cls = KFConfig
    n_report = 2                  # the reconciliation stage acts on (m1, m2)
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float | None, ...] = ()

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg=KFConfig()):
        self.dt = dt
        self.u = u_cmd
        self.cfg = cfg
        self.Q = self.process_noise(cfg, dt)
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = np.eye(2) * float(p0_std) ** 2
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    @staticmethod
    def process_noise(cfg, dt: float) -> np.ndarray:
        return np.eye(2) * cfg.sigma_w ** 2

    def drift(self, k: int) -> np.ndarray:
        """Deterministic state change over step k -> k + 1: the COMMANDED transfer."""
        return B * self.u[k] * self.dt

    def predict(self, x, P, k):
        return x + self.drift(k), P + self.Q

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        """Feedback: replace the state at the last ingested sampling step with (x, P).

        Used by the runner to push a projected (x*, P*) back into the filter. P may be
        rank-deficient (a hard projection sets A P* A^T = 0); the next predict adds Q.
        Refuses before anything has been ingested: the declared prior is an input and
        is never overwritten."""
        if not self.xf:
            raise ValueError("set_state before the first ingest would overwrite the declared prior")
        x = np.array(x, dtype=float, copy=True).reshape(2)
        P = np.array(P, dtype=float, copy=True).reshape(2, 2)
        self._x = x
        self._P = 0.5 * (P + P.T)
        self.xf[-1] = self._x.copy()
        self.Pf[-1] = self._P.copy()

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.xf):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, expected {len(self.xf)}")
        if j > 0:
            self._x, self._P = self.predict(self._x, self._P, j - 1)
        m = obs.mask
        nu_full = np.full(obs.y.size, np.nan)
        s_full = np.full(obs.y.size, np.nan)
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


class ClosedQKalmanFilter(KalmanFilter):
    """The same KF with closure encoded in the process noise instead of in a
    constraint row:

        Q = sigma_q^2 dt^2 B B^T + eps I,   B = (-1, +1)

    Process noise then moves mass between the reservoirs (the pump's declared
    fluctuation, sigma_q = 0.01 kg/s) and leaves the sum alone up to eps, so the
    filter's own uncertainty along the sum direction shrinks as the two sensors are
    averaged and never regrows. It has no constraint row, does not know the declared
    total b, and has nothing to stop enforcing when the boundary opens: a stale
    closure assumption makes it confidently wrong exactly as hard projection is.
    Its null-direction process noise (sqrt(2) sigma_q = 0.014 kg per step) is also
    smaller than KFConfig's untuned 0.05, so it averages longer and lags any real
    change in the difference direction more. Same declared prior and H as
    KalmanFilter."""

    model_version = "reservoir2-linear-closedq-v1"
    config_cls = ClosedQConfig

    @staticmethod
    def process_noise(cfg, dt: float) -> np.ndarray:
        return cfg.sigma_q ** 2 * dt ** 2 * np.outer(B, B) + cfg.eps * np.eye(2)


class OracleKalmanFilter(KalmanFilter):
    """ORACLE (BOUND), NOT A CANDIDATE. A KF whose deterministic drift uses the
    HIDDEN actual pump parameter rate and the hidden leak:

        x_{k+1} = x_k + B u_actual_k dt + (0, -leak_k) dt,   Q = eps I

    It is what a perfect model of the inputs would give, and only that: the
    per-step pump fluctuation and the valve transfer are realised in the truth as
    process noise the oracle does not model (Q = eps I only), so its bound is
    tight where the truth is deterministic given the inputs and it is over-
    confident where it is not (the noisy valve). The hidden arrays reach it only
    through `oracle_inputs`: experiment code passes them to runner.run() as its
    explicit `oracle_inputs` keyword, run() forwards them for spec.kind == "oracle"
    and raises for any other kind, and no other estimator accepts them."""

    model_version = "reservoir2-oracle-v1"
    config_cls = OracleConfig

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg=OracleConfig(),
                 oracle_inputs: tuple[np.ndarray, np.ndarray] | None = None):
        if oracle_inputs is None:
            raise ValueError("the oracle needs (u_actual, leak) as oracle_inputs; runner.run() forwards them "
                             "for kind 'oracle' only, and only when its caller passes them explicitly")
        super().__init__(x0, p0_std, dt, u_cmd, cfg)
        u_actual, leak = oracle_inputs
        self.u_actual = np.asarray(u_actual, dtype=float).copy()
        self.leak = np.asarray(leak, dtype=float).copy()
        if self.u_actual.shape != np.shape(u_cmd) or self.leak.shape != np.shape(u_cmd):
            raise ValueError("oracle inputs must be per-step arrays matching the commanded input")

    @staticmethod
    def process_noise(cfg, dt: float) -> np.ndarray:
        return cfg.eps * np.eye(2)

    def drift(self, k: int) -> np.ndarray:
        return B * self.u_actual[k] * self.dt + LEAK_DIR * self.leak[k] * self.dt


class HoldLast:
    """Baseline: hold the last value each sensor reported. Ignores the dynamics.
    Variance grows by sigma_w^2 per blind step so its intervals are at least honest
    about staleness."""

    model_version = "hold-last-value-v0"
    config_cls = KFConfig
    n_report = 2
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float | None, ...] = ()

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
        nan_row = np.full(obs.y.size, np.nan)
        self.innov.append(nan_row.copy())
        self.innov_var.append(nan_row.copy())
        self.innov_z.append(nan_row.copy())

    def report(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        j = len(self.hist) - 1
        if j < 0:
            return self.x0.copy(), np.diag(self.var0 + k * self.inflate)
        x, var = self.hist[j]
        return x.copy(), np.diag(var + (k - j) * self.inflate)

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        raise NotImplementedError("hold_last has no state to feed a projection back into")


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
    n_report = 2               # the reconciliation stage acts on the mass marginal only
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
        nu_full = np.full(obs.y.size, np.nan)
        s_full = np.full(obs.y.size, np.nan)
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

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        raise NotImplementedError(
            "feedback into the mass marginal of an augmented state is not implemented: replacing "
            "(x[:2], P[:2, :2]) alone would leave the cross-covariances with alpha and L inconsistent"
        )


ESTIMATORS = {
    "kf": KalmanFilter,
    "hold_last": HoldLast,
    "kf_aug": AugmentedKalmanFilter,
    "kf_closedq": ClosedQKalmanFilter,
    "oracle": OracleKalmanFilter,          # BOUND, not a candidate; see runner.run()
    # one tide-gauge series, n_report = 1 (the water level); see estimators_water
    "level_trend": LevelTrendKF,
    "tide_kf": TideKF,
    # one reservoir, four gauges; see estimators_balance
    "wb_open": WaterBalanceOpen,
    "wb_aug": WaterBalanceAugmented,
    "wb_closed": WaterBalanceClosed,
}
