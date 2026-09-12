"""Water-level estimators for ONE tide-gauge series: the first estimators in the tree that
model something real. Each takes a single sensor (one water-level column, as the DAF bridge
builds it: one (station, datum, unit) group) and reports one reconciled component, the
water level (n_report = 1); nothing is commanded in a tide-gauge record, so the public
input u must be zero throughout, and both refuse a record that says otherwise.

Both are linear Kalman filters with the interface and the innovation record of
estimators.KalmanFilter (ingest / report, innov / innov_var / innov_z per ingested
sampling step, reporting by predicting forward from the last ingested step), and both
take their time step from the PublicInputs grid: the runner hands them the clock
(`needs_clock`), dt = t[1] - t[0], and a clock that is not uniform to 1e-9 s is refused.
Time is in seconds, levels in the series' unit (m for the NOAA files in data/daf).

    level_trend   x = [level, rate]: the continuous white-noise-acceleration model
                      F = [[1, dt], [0, 1]],
                      Q = q_scale^2 [[dt^3/3, dt^2/2], [dt^2/2, dt]],
                  q_scale in m s^-3/2 (q_scale^2 is the acceleration's spectral density),
                  H = [1, 0]. rate is an extra state with nominal None (recorded, never
                  flagged).
    tide_kf       x = [mean level, a_M2, b_M2, a_K1, b_K1, a_O1, b_O1, a_M4, b_M4]:
                  harmonic regression with a random walk on every coefficient,
                      F = I,   Q = q_scale^2 dt I,   q_scale in m s^-1/2,
                      H_k = [1, cos(w_M2 t_k), sin(w_M2 t_k), ..., cos(w_M4 t_k), sin(w_M4 t_k)],
                  with t_k the record's own clock (seconds from its first grid point; the
                  phases are relative to that epoch, which the isotropic zero-mean prior on
                  each (a_c, b_c) pair makes immaterial). It reports the water level
                  H_k x_k first, then the nine states (aug_names, every nominal None), so
                  the reported covariance T P T^T (T = [H_k; I]) has rank 9 of 10 -- the
                  level is a linear function of the states, not a tenth state.

Constituent angular speeds, in degrees per mean solar hour, from Schureman, P. (1958), "Manual
of Harmonic Analysis and Prediction of Tides", U.S. Coast and Geodetic Survey Special
Publication No. 98 (the constituent speed table; NOAA CO-OPS lists the same speeds with each
station's harmonic constituents):

    M2 28.9841042 (principal lunar semidiurnal)   K1 15.0410686 (lunisolar diurnal)
    O1 13.9430356 (lunar diurnal)                 M4 57.9682084 (shallow-water overtide of M2)

What one day cannot resolve. Two constituents separate only over a record longer than the
Rayleigh period 360 / |speed difference| hours: S2 (30.0000000 deg/h) from M2 needs 14.8
days and N2 (28.4397295 deg/h) from M2 27.6 days, so one day of data cannot separate S2 or
N2 from M2 -- both are absorbed into (a_M2, b_M2); K1 from O1 needs 13.7 days, so on one day
only their sum is resolved and the individual K1 / O1 coefficients are not interpretable;
even M2 against K1 (25.8 h) and against O1 (23.9 h) sits at the one-day limit
(rayleigh_period_hours). The coefficients are nuisance states that let the filter predict the
next six minutes, not a harmonic analysis; nothing in the tree reads them as constituent
amplitudes.

Declared priors (stated, wide, never fitted): the level (level_trend) or the mean level
(tide_kf) comes from run()'s declared prior, (level0,) and level0_std -- the experiment
declares 0 m with a 10 m std, over three times the largest reading in data/daf (2.775 m, on
the STND datum); the rest from the configs below: rate ~ N(0, (1e-3 m/s)^2), i.e. 3.6 m/h, over
four times the fastest six-minute change in data/daf (0.083 m in six minutes, 0.83 m/h), and
each harmonic coefficient ~ N(0, (2 m)^2), twice the half-range of the widest day in data/daf
(-0.204 to 1.711 m MLLW on 2024-01-15). The only free parameter is q_scale, which has no
default: it is identified by experiments.real_noaa on a declared window and evaluated on
another.

Neither filter accepts feedback (set_state raises) and neither has anything to flag: a single
series declares no constraint, so the reconciliation stage skips every step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schema import Observation

# Degrees per mean solar hour (Schureman 1958, SP 98); see the module docstring.
CONSTITUENT_SPEED_DEG_PER_HOUR: dict[str, float] = {
    "M2": 28.9841042,
    "K1": 15.0410686,
    "O1": 13.9430356,
    "M4": 57.9682084,
}
TIDE_CONSTITUENTS: tuple[str, ...] = ("M2", "K1", "O1", "M4")
# Not modelled: the two semidiurnal constituents one day cannot separate from M2 (same source).
UNRESOLVED_ON_ONE_DAY_DEG_PER_HOUR: dict[str, float] = {"S2": 30.0000000, "N2": 28.4397295}
CLOCK_TOLERANCE_S = 1e-9


def speed_rad_per_s(name: str) -> float:
    return CONSTITUENT_SPEED_DEG_PER_HOUR[name] * math.pi / 180.0 / 3600.0


def rayleigh_period_hours(a: str, b: str) -> float:
    """The record length, in hours, over which constituents a and b separate: 360 / |speed_a - speed_b|."""
    speeds = {**CONSTITUENT_SPEED_DEG_PER_HOUR, **UNRESOLVED_ON_ONE_DAY_DEG_PER_HOUR}
    return 360.0 / abs(speeds[a] - speeds[b])


def _check_q_scale(q_scale) -> float:
    if isinstance(q_scale, bool) or not isinstance(q_scale, (int, float)) or not math.isfinite(q_scale) \
            or q_scale <= 0.0:
        raise ValueError(f"q_scale must be a finite positive number; got {q_scale!r}")
    return float(q_scale)


@dataclass(frozen=True)
class LevelTrendConfig:
    """level_trend. q_scale (m s^-3/2) has no default: it is identified on a declared window."""
    q_scale: float
    rate0: float = 0.0          # declared prior mean of the rate, m/s
    rate0_std: float = 1e-3     # m/s (3.6 m/h): wide

    def __post_init__(self):
        _check_q_scale(self.q_scale)
        if not (math.isfinite(self.rate0) and math.isfinite(self.rate0_std) and self.rate0_std > 0.0):
            raise ValueError("rate0 must be finite and rate0_std finite and positive")


@dataclass(frozen=True)
class TideConfig:
    """tide_kf. q_scale (m s^-1/2) has no default: it is identified on a declared window.
    Every harmonic coefficient's prior is N(0, coef0_std^2)."""
    q_scale: float
    coef0_std: float = 2.0      # m: wide

    def __post_init__(self):
        _check_q_scale(self.q_scale)
        if not (math.isfinite(self.coef0_std) and self.coef0_std > 0.0):
            raise ValueError("coef0_std must be finite and positive")


class _SingleSeriesKF:
    """Shared machinery: a linear KF over one sensor with a per-step observation row H(j),
    a constant F and Q, the Joseph update, the innovation record and forward reporting.
    Subclasses set the state, the prior, F, Q, H(j) and the reported transform."""

    n_report = 1                # the reconciled component: the water level
    needs_clock = True          # runner.run() passes clock=inputs.t
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float | None, ...] = ()

    def __init__(self, x0, p0: np.ndarray, F: np.ndarray, Q: np.ndarray, clock, u_cmd):
        t = np.asarray(clock, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("a water-level filter needs the record's clock (at least two steps)")
        dt = float(t[1] - t[0])
        if not np.all(np.abs(np.diff(t) - dt) <= CLOCK_TOLERANCE_S):
            raise ValueError("the clock is not uniform: a water-level filter takes one dt from the grid")
        if np.any(np.asarray(u_cmd, dtype=float) != 0.0):
            raise ValueError("a tide-gauge record commands nothing; u_commanded must be zero throughout")
        self.t = t.copy()
        self.dt = dt
        self.F = F
        self.Q = Q
        self.N = F.shape[0]
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = p0
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    def H(self, j: int) -> np.ndarray:          # (N,) observation row at sampling step j
        raise NotImplementedError

    def reported(self, x: np.ndarray, P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return x, P

    def predict(self, x, P):
        return self.F @ x, self.F @ P @ self.F.T + self.Q

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.xf):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, expected {len(self.xf)}")
        if np.asarray(obs.y).size != 1:
            raise ValueError(f"{type(self).__name__} reads one series; this observation carries "
                             f"{np.asarray(obs.y).size} (two datums or two stations are two quantities, never one)")
        if j > 0:
            self._x, self._P = self.predict(self._x, self._P)
        nu_full = np.full(1, np.nan)
        s_full = np.full(1, np.nan)
        if bool(obs.mask[0]):
            h = self.H(j)
            r = float(obs.R[0, 0])
            ph = self._P @ h
            s = float(h @ ph) + r
            k_gain = ph / s
            nu = float(obs.y[0]) - float(h @ self._x)         # innovation against the prediction
            self._x = self._x + k_gain * nu
            I_KH = np.eye(self.N) - np.outer(k_gain, h)
            P = I_KH @ self._P @ I_KH.T + r * np.outer(k_gain, k_gain)   # Joseph form
            self._P = 0.5 * (P + P.T)
            nu_full[0] = nu
            s_full[0] = s
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
        for _ in range(start, k):
            x, P = self.predict(x, P)
        return self.reported(x, P, k)

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        raise NotImplementedError(f"{type(self).__name__}: a single series declares no constraint; "
                                  "there is no projection to feed back")


class LevelTrendKF(_SingleSeriesKF):
    """x = [level, rate] under continuous white-noise acceleration; see the module docstring."""

    model_version = "water-level-trend-cwna-v1"
    config_cls = LevelTrendConfig
    aug_names = ("rate",)
    aug_nominal = (None,)

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: LevelTrendConfig, *, clock):
        level0 = np.asarray(x0, dtype=float).reshape(-1)
        if level0.size != 1:
            raise ValueError(f"level_trend's declared prior is one level; got {level0.size} values")
        d = float(np.asarray(clock, dtype=float)[1] - np.asarray(clock, dtype=float)[0])
        q2 = cfg.q_scale ** 2
        F = np.array([[1.0, d], [0.0, 1.0]])
        Q = q2 * np.array([[d ** 3 / 3.0, d ** 2 / 2.0], [d ** 2 / 2.0, d]])
        P0 = np.diag([float(p0_std) ** 2, cfg.rate0_std ** 2])
        super().__init__([level0[0], cfg.rate0], P0, F, Q, clock, u_cmd)
        if abs(self.dt - float(dt)) > CLOCK_TOLERANCE_S:
            raise ValueError(f"dt {dt} disagrees with the clock's {self.dt}")
        self.cfg = cfg
        self._h = np.array([1.0, 0.0])

    def H(self, j: int) -> np.ndarray:
        return self._h


class TideKF(_SingleSeriesKF):
    """x = [mean level, (a_c, b_c) for M2, K1, O1, M4] with a random walk on every state;
    reports [H_k x, x]; see the module docstring."""

    model_version = "water-level-harmonic-rw-m2k1o1m4-v1"
    config_cls = TideConfig
    constituents = TIDE_CONSTITUENTS
    aug_names = ("mean_level",) + tuple(f"{c}_{ab}" for c in TIDE_CONSTITUENTS for ab in ("a", "b"))
    aug_nominal = (None,) * (1 + 2 * len(TIDE_CONSTITUENTS))

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: TideConfig, *, clock):
        level0 = np.asarray(x0, dtype=float).reshape(-1)
        if level0.size != 1:
            raise ValueError(f"tide_kf's declared prior is one mean level; got {level0.size} values")
        n = 1 + 2 * len(self.constituents)
        d = float(np.asarray(clock, dtype=float)[1] - np.asarray(clock, dtype=float)[0])
        F = np.eye(n)
        Q = cfg.q_scale ** 2 * d * np.eye(n)
        P0 = np.diag([float(p0_std) ** 2] + [cfg.coef0_std ** 2] * (n - 1))
        super().__init__([level0[0]] + [0.0] * (n - 1), P0, F, Q, clock, u_cmd)
        if abs(self.dt - float(dt)) > CLOCK_TOLERANCE_S:
            raise ValueError(f"dt {dt} disagrees with the clock's {self.dt}")
        self.cfg = cfg
        w = np.array([speed_rad_per_s(c) for c in self.constituents])
        phase = np.outer(self.t, w)                             # (steps, constituents)
        rows = np.ones((self.t.size, n))
        rows[:, 1::2] = np.cos(phase)
        rows[:, 2::2] = np.sin(phase)
        self._rows = rows

    def H(self, j: int) -> np.ndarray:
        return self._rows[j]

    def reported(self, x: np.ndarray, P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        h = self._rows[k]
        T = np.vstack([h, np.eye(self.N)])                      # (1 + N, N): level, then the states
        return T @ x, T @ P @ T.T
