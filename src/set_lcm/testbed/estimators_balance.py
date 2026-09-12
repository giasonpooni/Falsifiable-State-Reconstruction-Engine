"""Reservoir water-balance filters: the first estimators whose constraint is a real
conservation relation over evidence from four independent gauges.

The system is one control volume -- a reservoir -- with a measured storage, a measured
outflow below the dam, and two measured inflows. Over a day, conservation says

    dS/dt = (inflow) - (outflow) + (ungauged net inflow)

and the ungauged term is not zero: at Ridgway the gauged drainage is 246.2 of 265 sq mi,
so 7.1% of the catchment is unmeasured, and evaporation, precipitation on the lake and any
gauge bias all land in the same term. That is the point. The declared constraint here is
the statement that the gauges close the balance, and the consistency statistic is what
tests it.

Units are the sources' own, converted nowhere but here, once, explicitly: storage in
acre-feet, flows in cubic feet per second, time in days. One cubic foot per second
sustained for one day is

    CFS_DAY_TO_ACRE_FT = 86400 s/day / 43560 ft^2 per acre = 1.9834710743801653 acre-ft

(43560 ft^2 in an acre is exact by the definition of the international foot and the
survey-derived acre; the ratio above is exact in binary only to the printed digits).

THE THREE FILTERS, and what each one assumes

    wb_open    x = [S, G, qout, qin1, qin2],  n_report = 2 (S and G)
               S  the storage, a RANDOM WALK: the filter is told nothing about why storage
                  moves, only what the storage gauge says.
               G  the cumulative gauged net inflow volume since the epoch, an exact
                  integral of the three rate states: G' = G + c dt (qin1 + qin2 - qout).
               Each rate is a random walk measured by its own gauge.
               Nothing links S to G except the DECLARED CONSTRAINT  S - G = S0. So the
               consistency statistic here is exactly the reconciliation question: do two
               independent measurement chains -- a storage gauge, and three flow gauges
               integrated -- agree, to within what their own uncertainties allow? When they
               do not, that is a real imbalance, and the kernel says so instead of averaging
               it away.

    wb_aug     x = [S, G, U, qout, qin1, qin2],  n_report = 3 (S, G and U)
               U is the cumulative UNGAUGED net inflow volume, a random walk with its own
               declared q. The constraint becomes S - G - U = S0. Some value of U can
               always satisfy that equation, but its finite prior and process variance
               still allow the preprojection statistic to reject sufficiently large
               disagreement. U is unchanged by gauge observations alone; reconciliation
               estimates a modeled imbalance, and feedback carries that estimate forward.
               U_hat does not distinguish ungauged water from gauge or model errors.

    wb_closed  x = [S, qout, qin1, qin2],  n_report = 1 (S)
               Closure in the DYNAMICS: S' = S + c dt (qin1 + qin2 - qout). The filter
               assumes the gauges close the balance rather than being told it as a
               constraint, which is what a modeller who trusted the gauges would write. It
               declares no constraint, so nothing can reject it on this statistic; it is
               here as the baseline that shows what assuming closure costs, the analogue of
               kf_closedq.

WHAT IS DECLARED, NEVER FITTED. Every prior width and process-noise scale is a declared
number in `BalanceConfig`, in the units above, and the experiment states each one with the
reasoning behind it. R comes from the bridge -- and for USGS daily values the source states
NO uncertainty, so R is consumer-declared with a citation and the report says so in those
words.

THE CONSTRAINT'S OWN UNCERTAINTY. b = S0 is a storage READING, not a known constant, so the
constraint set declares `b_var` = the variance of that reading. This is the first use of
b_var on real evidence: an exact b would claim the balance is anchored to a number nobody
measured exactly.

WHAT THE BALANCE CANNOT SEE, structurally. `detectability(f)` is zero along null(A). For
wb_open's A = [1, -1] that is the direction [1, 1]/sqrt(2): storage and cumulative gauged
inflow rising together. A bias that inflates the storage reading and the net gauged inflow
by the same volume is invisible to this constraint however large it is. Worse and more
physical: an EQUAL bias on an inflow gauge and the outflow gauge cancels in
(qin1 + qin2 - qout) before it ever reaches the state, so it is invisible to the balance
and to d(f) alike -- it is not a direction in state space at all. The experiment reports
both, because only the first is what d(f) measures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Observation

__all__ = [
    "CFS_DAY_TO_ACRE_FT", "SECONDS_PER_DAY", "BalanceConfig", "WaterBalanceOpen",
    "WaterBalanceAugmented", "WaterBalanceClosed",
]

SECONDS_PER_DAY = 86400.0
SQ_FT_PER_ACRE = 43560.0
CFS_DAY_TO_ACRE_FT = SECONDS_PER_DAY / SQ_FT_PER_ACRE
CLOCK_TOLERANCE_S = 1e-9

# sensor column order, as the experiment declares its selectors
S_COL, QOUT_COL, QIN1_COL, QIN2_COL = 0, 1, 2, 3
N_SENSORS = 4


@dataclass(frozen=True)
class BalanceConfig:
    """Declared, never fitted. Volumes in acre-ft, rates in ft^3/s, time in days.

    `q_storage` and `q_flow` are random-walk scales per sqrt(day): a state's standard
    deviation grows by that much in one day when nothing measures it. `q_ungauged` is
    wb_aug's only extra freedom -- how fast the unmeasured term is allowed to move.
    """
    q_storage: float            # acre-ft / sqrt(day), on S
    q_flow: float               # ft^3/s / sqrt(day), on each rate state
    q_ungauged: float = 0.0     # acre-ft / sqrt(day), on U (wb_aug only)
    storage0_std: float = 0.0   # prior sd on S, acre-ft; 0 means "use run()'s p0_std"
    cumulative0_std: float = 1.0    # prior sd on G and U at the epoch, acre-ft
    flow0: float = 0.0          # prior mean on each rate state, ft^3/s
    flow0_std: float = 1000.0   # prior sd on each rate state, ft^3/s

    def __post_init__(self):
        for name in ("q_storage", "q_flow", "q_ungauged", "storage0_std", "cumulative0_std", "flow0_std"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0.0:
                raise ValueError(f"BalanceConfig.{name} must be finite and >= 0; got {v!r}")
        if self.q_flow <= 0.0 or self.flow0_std <= 0.0 or self.cumulative0_std <= 0.0:
            raise ValueError("q_flow, flow0_std and cumulative0_std must be strictly positive: a state "
                             "with no uncertainty anywhere cannot be updated")


class _BalanceKF:
    """A linear KF over four gauges with a constant F, Q and H, the Joseph update per
    sensor, the innovation record, and forward reporting -- the same contract as
    estimators_water._SingleSeriesKF, for four sensors instead of one.

    Sensors are updated one at a time (sequential scalar updates, which is exact for a
    diagonal R and is what the bridge always produces), so a day with some gauges reporting
    and others missing needs no special case: a masked sensor is simply not applied, and its
    innovation is NaN.
    """

    needs_clock = True
    aug_names: tuple[str, ...] = ()
    aug_nominal: tuple[float | None, ...] = ()

    def __init__(self, x0, P0: np.ndarray, F: np.ndarray, Q: np.ndarray, H: np.ndarray, clock, u_cmd):
        t = np.asarray(clock, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("a water-balance filter needs the record's clock (at least two steps)")
        dt_s = float(t[1] - t[0])
        if not np.all(np.abs(np.diff(t) - dt_s) <= CLOCK_TOLERANCE_S):
            raise ValueError("the clock is not uniform: a water-balance filter takes one dt from the grid")
        if np.any(np.asarray(u_cmd, dtype=float) != 0.0):
            raise ValueError("a gauge record commands nothing; u_commanded must be zero throughout")
        self.t = t.copy()
        self.dt = dt_s
        self.dt_days = dt_s / SECONDS_PER_DAY
        self.F, self.Q, self._H = F, Q, H
        self.N = F.shape[0]
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.P0 = P0
        self._x = self.x0.copy()
        self._P = self.P0.copy()
        self.xf: list[np.ndarray] = []
        self.Pf: list[np.ndarray] = []
        self.innov: list[np.ndarray] = []
        self.innov_var: list[np.ndarray] = []
        self.innov_z: list[np.ndarray] = []

    def H(self, j: int) -> np.ndarray:
        return self._H

    def reported(self, x: np.ndarray, P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return x, P

    def predict(self, x, P):
        return self.F @ x, self.F @ P @ self.F.T + self.Q

    def ingest(self, obs: Observation, j: int) -> None:
        if j != len(self.xf):
            raise ValueError(f"observations must be ingested in sampling order: got step {j}, "
                             f"expected {len(self.xf)}")
        y = np.asarray(obs.y, dtype=float)
        if y.size != N_SENSORS:
            raise ValueError(f"{type(self).__name__} reads {N_SENSORS} gauges (storage, outflow, two "
                             f"inflows); this observation carries {y.size}")
        if j > 0:
            self._x, self._P = self.predict(self._x, self._P)
        Hm = self.H(j)
        nu_full = np.full(N_SENSORS, np.nan)
        s_full = np.full(N_SENSORS, np.nan)
        for i in range(N_SENSORS):
            if not bool(obs.mask[i]):
                continue
            h = Hm[i]
            r = float(obs.R[i, i])
            ph = self._P @ h
            s = float(h @ ph) + r
            k_gain = ph / s
            nu = float(y[i]) - float(h @ self._x)
            self._x = self._x + k_gain * nu
            I_KH = np.eye(self.N) - np.outer(k_gain, h)
            P = I_KH @ self._P @ I_KH.T + r * np.outer(k_gain, k_gain)    # Joseph form
            self._P = 0.5 * (P + P.T)
            nu_full[i] = nu
            s_full[i] = s
        self.xf.append(self._x.copy())
        self.Pf.append(self._P.copy())
        self.innov.append(nu_full)
        self.innov_var.append(s_full)
        self.innov_z.append(nu_full / np.sqrt(s_full))

    def set_state(self, x_report: np.ndarray, P_report: np.ndarray) -> None:
        """Replace the reported marginal, preserving the rates' conditional distribution.

        Requires an ingested sampling step; the declared initial prior is never replaced.

        For reported variables r and remaining rate states z, the prior conditional is
        z | r ~ N(mu_z + B (r - mu_r), C), where B = P_zr P_rr^-1. Replacing the
        marginal of r therefore also changes the rate means, their covariance, and the
        cross-covariance. When (x_report, P_report) comes from a Gaussian constraint
        update on r, this is the same posterior as applying that update to the full
        state with zero constraint coefficients on z. Copying just the marginal block
        would violate those correlations and can make the full covariance indefinite.

        Only used under EstimatorSpec(feedback=True), and only for the balance filters,
        where it is the difference between an augmented state that is estimated and one
        that is not: U is observed by NOTHING except the constraint, so without feedback it
        stays at its prior for ever and only the reported output moves. The repository's
        standing warning applies in full -- a constraint fed back is absorbed as if it were
        fresh evidence, which changes the statistic that tests it. Updating the full
        covariance does not make repeated use of the same uncertain reference independent.
        """
        if not self.xf:
            raise ValueError("set_state before the first ingest would overwrite the declared prior")
        n = int(self.n_report)
        x = np.asarray(x_report, dtype=float).reshape(-1)
        P = np.asarray(P_report, dtype=float)
        if x.size != n or P.shape != (n, n):
            raise ValueError(f"set_state expects the first {n} reported component(s); got x {x.shape} "
                             f"and P {P.shape}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(P))):
            raise ValueError("set_state requires finite state and covariance")
        if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
            raise ValueError("set_state requires a symmetric covariance")
        P = 0.5 * (P + P.T)
        scale = max(1.0, float(np.linalg.norm(P, ord=2)))
        if np.linalg.eigvalsh(P)[0] < -1e-12 * scale:
            raise ValueError("set_state requires a positive-semidefinite covariance")

        prior_P = self._P
        try:
            B = np.linalg.solve(prior_P[:n, :n], prior_P[:n, n:]).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("set_state requires a nonsingular prior reported covariance") from exc
        # Form the conditional covariance as a congruence of the full prior instead of
        # subtracting covariance blocks. The assembled posterior is a sum of congruences.
        conditional_map = np.hstack((-B, np.eye(self.N - n)))
        conditional_P = conditional_map @ prior_P @ conditional_map.T
        marginal_map = np.vstack((np.eye(n), B))
        full_P = marginal_map @ P @ marginal_map.T
        full_P[n:, n:] += conditional_P
        full_x = np.concatenate((x, self._x[n:] + B @ (x - self._x[:n])))
        self._x = full_x
        self._P = 0.5 * (full_P + full_P.T)
        self.xf[-1] = self._x.copy()
        self.Pf[-1] = self._P.copy()

    def report(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict forward from the last ingested step to report step k, as every estimator
        in the tree does: an estimate for a step whose evidence has not arrived is a
        prediction, and says so through its covariance."""
        j = len(self.xf) - 1
        if j < 0:
            x, P, start = self.x0.copy(), self.P0.copy(), 0
        else:
            x, P, start = self.xf[j].copy(), self.Pf[j].copy(), j
        for _ in range(max(0, k - start)):
            x, P = self.predict(x, P)
        return self.reported(x, P, k)


class WaterBalanceOpen(_BalanceKF):
    """x = [S, G, qout, qin1, qin2]; reports [S, G] and the three rates. See the module
    docstring: S and G are linked only by the declared constraint."""

    model_version = "reservoir-balance-open-v1"
    config_cls = BalanceConfig
    n_report = 2
    aug_names = ("qout", "qin1", "qin2")
    aug_nominal = (None, None, None)

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: BalanceConfig, *, clock):
        s0 = np.asarray(x0, dtype=float).reshape(-1)
        if s0.size != 1:
            raise ValueError(f"{type(self).__name__}'s declared prior is one storage; got {s0.size} values")
        t = np.asarray(clock, dtype=float)
        dt_days = float(t[1] - t[0]) / SECONDS_PER_DAY
        c = CFS_DAY_TO_ACRE_FT * dt_days
        F = np.eye(5)
        F[1, 2:5] = [-c, c, c]                      # G' = G + c dt (qin1 + qin2 - qout)
        q_s = cfg.q_storage ** 2 * dt_days
        q_f = cfg.q_flow ** 2 * dt_days
        Q = np.diag([q_s, 0.0, q_f, q_f, q_f])
        sd_s = cfg.storage0_std or float(p0_std)
        P0 = np.diag([sd_s ** 2, cfg.cumulative0_std ** 2,
                      cfg.flow0_std ** 2, cfg.flow0_std ** 2, cfg.flow0_std ** 2])
        H = np.zeros((N_SENSORS, 5))
        H[S_COL, 0] = 1.0
        H[QOUT_COL, 2] = 1.0
        H[QIN1_COL, 3] = 1.0
        H[QIN2_COL, 4] = 1.0
        super().__init__([s0[0], 0.0, cfg.flow0, cfg.flow0, cfg.flow0], P0, F, Q, H, clock, u_cmd)
        if abs(self.dt - float(dt)) > CLOCK_TOLERANCE_S:
            raise ValueError(f"dt {dt} disagrees with the clock's {self.dt}")
        self.cfg = cfg


class WaterBalanceAugmented(_BalanceKF):
    """x = [S, G, U, qout, qin1, qin2]; reports [S, G, U] and the three rates. U is the
    cumulative ungauged net inflow the constraint cannot otherwise account for."""

    model_version = "reservoir-balance-augmented-v1"
    config_cls = BalanceConfig
    n_report = 3
    aug_names = ("qout", "qin1", "qin2")
    aug_nominal = (None, None, None)

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: BalanceConfig, *, clock):
        s0 = np.asarray(x0, dtype=float).reshape(-1)
        if s0.size != 1:
            raise ValueError(f"{type(self).__name__}'s declared prior is one storage; got {s0.size} values")
        if cfg.q_ungauged <= 0.0:
            raise ValueError("wb_aug needs a positive q_ungauged: a cumulative ungauged term that cannot "
                             "move is not an augmented state, it is the open filter with an extra zero")
        t = np.asarray(clock, dtype=float)
        dt_days = float(t[1] - t[0]) / SECONDS_PER_DAY
        c = CFS_DAY_TO_ACRE_FT * dt_days
        F = np.eye(6)
        F[1, 3:6] = [-c, c, c]
        q_s = cfg.q_storage ** 2 * dt_days
        q_u = cfg.q_ungauged ** 2 * dt_days
        q_f = cfg.q_flow ** 2 * dt_days
        Q = np.diag([q_s, 0.0, q_u, q_f, q_f, q_f])
        sd_s = cfg.storage0_std or float(p0_std)
        P0 = np.diag([sd_s ** 2, cfg.cumulative0_std ** 2, cfg.cumulative0_std ** 2,
                      cfg.flow0_std ** 2, cfg.flow0_std ** 2, cfg.flow0_std ** 2])
        H = np.zeros((N_SENSORS, 6))
        H[S_COL, 0] = 1.0
        H[QOUT_COL, 3] = 1.0
        H[QIN1_COL, 4] = 1.0
        H[QIN2_COL, 5] = 1.0
        super().__init__([s0[0], 0.0, 0.0, cfg.flow0, cfg.flow0, cfg.flow0], P0, F, Q, H, clock, u_cmd)
        if abs(self.dt - float(dt)) > CLOCK_TOLERANCE_S:
            raise ValueError(f"dt {dt} disagrees with the clock's {self.dt}")
        self.cfg = cfg


class WaterBalanceClosed(_BalanceKF):
    """x = [S, qout, qin1, qin2]; reports [S] and the three rates. Closure is in the
    dynamics: this filter ASSUMES the gauges close the balance, so nothing can reject it on
    the consistency statistic. The baseline, not a candidate."""

    model_version = "reservoir-balance-closed-v1"
    config_cls = BalanceConfig
    n_report = 1
    aug_names = ("qout", "qin1", "qin2")
    aug_nominal = (None, None, None)

    def __init__(self, x0, p0_std: float, dt: float, u_cmd: np.ndarray, cfg: BalanceConfig, *, clock):
        s0 = np.asarray(x0, dtype=float).reshape(-1)
        if s0.size != 1:
            raise ValueError(f"{type(self).__name__}'s declared prior is one storage; got {s0.size} values")
        t = np.asarray(clock, dtype=float)
        dt_days = float(t[1] - t[0]) / SECONDS_PER_DAY
        c = CFS_DAY_TO_ACRE_FT * dt_days
        F = np.eye(4)
        F[0, 1:4] = [-c, c, c]                      # S' = S + c dt (qin1 + qin2 - qout)
        q_s = cfg.q_storage ** 2 * dt_days
        q_f = cfg.q_flow ** 2 * dt_days
        Q = np.diag([q_s, q_f, q_f, q_f])
        sd_s = cfg.storage0_std or float(p0_std)
        P0 = np.diag([sd_s ** 2, cfg.flow0_std ** 2, cfg.flow0_std ** 2, cfg.flow0_std ** 2])
        H = np.zeros((N_SENSORS, 4))
        H[S_COL, 0] = 1.0
        H[QOUT_COL, 1] = 1.0
        H[QIN1_COL, 2] = 1.0
        H[QIN2_COL, 3] = 1.0
        super().__init__([s0[0], cfg.flow0, cfg.flow0, cfg.flow0], P0, F, Q, H, clock, u_cmd)
        if abs(self.dt - float(dt)) > CLOCK_TOLERANCE_S:
            raise ValueError(f"dt {dt} disagrees with the clock's {self.dt}")
        self.cfg = cfg
