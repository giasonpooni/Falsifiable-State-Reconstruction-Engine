"""A two-reach Muskingum river: the hidden truth, the readings it produces, and the rows.

Roadmap stage 3's second independent balance. The design study
(experiments/second_balance.py) measured what this topology can and cannot separate before it
was built; this is the system that lets those predictions be tested against simulated records
rather than asserted from the algebra.

WHY ROUTING AND NOT JUST A SECOND CONTINUITY ROW. Continuity differences the storage at the
two ends of an interval, so a PERSISTENT storage-sensor bias cancels out of it exactly: with
continuity alone that fault is structurally invisible and only 2 of 7 declared candidates are
isolable. Muskingum's own storage relation S = K[x I + (1 - x) O] averages the same two
storage readings instead of differencing them, so the bias survives, and the flows enter with
coefficients that differ from continuity's. That is what takes the topology to 5 of 7.

HIDDEN TRUTH STAYS HIDDEN. `simulate()` returns the true flows, storages and any injected
fault; `observe()` turns them into readings. Nothing on the estimator side receives a
`TwoReachTruth`, and the fault record exists so an evaluator can score, not so a detector can
peek. The declared constraint rows are built from DECLARED parameters, which a caller may set
to something other than the truth -- that is the parameter-error case, and it is the reason
ConstraintSet.A_var exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MuskingumConfig", "TwoReachTruth", "Readings", "Fault", "routing_coefficients",
           "route", "simulate", "observe", "constraint_rows", "design_matrix", "reading_names",
           "fault_reading_direction"]

SENSORS = ("storage_1", "storage_2", "inflow_gauge", "middle_gauge", "outflow_gauge")


@dataclass(frozen=True)
class MuskingumConfig:
    """Reach parameters and the sampling interval, in consistent time units.

    The standard stability window is 2 K x <= dt <= 2 K (1 - x); outside it the discrete
    routing coefficients change sign and the scheme oscillates. Validated rather than assumed
    so a declared-parameter sweep cannot wander out of it silently.
    """
    K: float = 1.0            # storage-time constant, days
    x: float = 0.2            # inflow weighting, 0 <= x <= 0.5
    dt: float = 1.0           # sampling interval, days

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.K, self.x, self.dt)):
            raise ValueError("Muskingum parameters must be finite")
        if self.K <= 0.0 or self.dt <= 0.0:
            raise ValueError("K and dt must be positive")
        if not 0.0 <= self.x <= 0.5:
            raise ValueError("the Muskingum weighting x must lie in [0, 0.5]")
        if not (2.0 * self.K * self.x <= self.dt <= 2.0 * self.K * (1.0 - self.x)):
            raise ValueError(
                f"dt = {self.dt:g} is outside the stability window "
                f"[{2 * self.K * self.x:g}, {2 * self.K * (1 - self.x):g}] for K = {self.K:g}, "
                f"x = {self.x:g}; the discrete routing coefficients would change sign")


@dataclass(frozen=True)
class Fault:
    """What was injected. Read by evaluators, never by anything on the estimator side.

    A SENSOR fault perturbs a reading and leaves the river alone. A PHYSICAL one changes the
    river and leaves the instruments honest. The declared balance cannot tell those apart when
    they produce the same residual direction, which is the point of carrying both.
    """
    kind: str                       # "none", "sensor_bias", "common_drift", "lateral_inflow"
    magnitude: float = 0.0
    onset: int = 0
    sensors: tuple[str, ...] = ()
    physical: bool = False

    def __post_init__(self):
        if self.kind not in ("none", "sensor_bias", "common_drift", "lateral_inflow"):
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if any(name not in SENSORS for name in self.sensors):
            raise ValueError(f"fault sensors must be drawn from {SENSORS}")


@dataclass(frozen=True)
class TwoReachTruth:
    """HIDDEN. The river as it actually behaved, plus what was done to it."""
    config: MuskingumConfig
    inflow: np.ndarray            # (n,) reach-1 inflow at each grid time
    middle: np.ndarray            # (n,) reach-1 outflow = reach-2 inflow
    outflow: np.ndarray           # (n,) reach-2 outflow
    storage_1: np.ndarray         # (n,)
    storage_2: np.ndarray         # (n,)
    lateral: np.ndarray           # (n,) ungauged inflow to reach 1; zero unless injected
    fault: Fault = field(default_factory=lambda: Fault("none"))


@dataclass(frozen=True)
class Readings:
    """What the estimator side receives: values, declared sigmas, and nothing else."""
    values: dict[str, np.ndarray]
    sigma: dict[str, float]
    n_steps: int

    def stacked(self) -> tuple[np.ndarray, np.ndarray]:
        """The reading vector in `reading_names` order, and its declared diagonal covariance."""
        names = reading_names(self.n_steps)
        y = np.array([self.values[sensor][index] for sensor, index in names])
        R = np.diag([self.sigma[sensor] ** 2 for sensor, _ in names])
        return y, R


def routing_coefficients(config: MuskingumConfig) -> tuple[float, float, float]:
    """(C0, C1, C2) of O_{k+1} = C0 I_{k+1} + C1 I_k + C2 O_k; they sum to one exactly."""
    K, x, dt = config.K, config.x, config.dt
    denominator = K - K * x + 0.5 * dt
    return ((-K * x + 0.5 * dt) / denominator,
            (K * x + 0.5 * dt) / denominator,
            (K - K * x - 0.5 * dt) / denominator)


def route(inflow: np.ndarray, config: MuskingumConfig) -> np.ndarray:
    """One reach, started from steady state so the record has no spin-up transient."""
    inflow = np.asarray(inflow, dtype=float)
    C0, C1, C2 = routing_coefficients(config)
    out = np.empty_like(inflow)
    out[0] = inflow[0]
    for k in range(inflow.size - 1):
        out[k + 1] = C0 * inflow[k + 1] + C1 * inflow[k] + C2 * out[k]
    return out


def flood_wave(n_steps: int, *, base: float = 10.0, peak: float = 40.0,
               rise: float = 8.0) -> np.ndarray:
    """A declared hydrograph. Deterministic: the randomness in this testbed is measurement
    noise and nothing else, so a seed changes the readings and never the river."""
    t = np.arange(n_steps, dtype=float)
    shape = (t / rise) ** 2 * np.exp(-(t / rise) + 1.0) / np.exp(1.0)
    return base + peak * shape


def simulate(config: MuskingumConfig, n_steps: int, fault: Fault | None = None) -> TwoReachTruth:
    """The river. A physical fault changes it here; a sensor fault does not reach it."""
    fault = fault or Fault("none")
    inflow = flood_wave(n_steps)
    lateral = np.zeros(n_steps)
    if fault.kind == "lateral_inflow":
        lateral[fault.onset:] = fault.magnitude
    middle = route(inflow + lateral, config)
    outflow = route(middle, config)
    storage_1 = config.K * (config.x * (inflow + lateral) + (1.0 - config.x) * middle)
    storage_2 = config.K * (config.x * middle + (1.0 - config.x) * outflow)
    return TwoReachTruth(config, inflow, middle, outflow, storage_1, storage_2, lateral, fault)


def observe(truth: TwoReachTruth, sigma: dict[str, float], rng: np.random.Generator) -> Readings:
    """Readings: the truth plus declared independent noise, plus any SENSOR fault.

    The interval-mean flow a gauge reports is the trapezoidal mean of the two grid values,
    which is what makes continuity exact on the noiseless record rather than approximate.
    """
    if set(sigma) != set(SENSORS):
        raise ValueError(f"declare a sigma for exactly {SENSORS}")
    if any(not np.isfinite(v) or v <= 0.0 for v in sigma.values()):
        raise ValueError("every declared sigma must be finite and positive")
    n = truth.inflow.size
    grid_inflow = truth.inflow + truth.lateral        # the gauge sees only what passes it
    mean_of = lambda series: 0.5 * (series[1:] + series[:-1])
    clean = {
        "storage_1": truth.storage_1.copy(),
        "storage_2": truth.storage_2.copy(),
        "inflow_gauge": mean_of(truth.inflow),        # the lateral inflow is UNGAUGED
        "middle_gauge": mean_of(truth.middle),
        "outflow_gauge": mean_of(truth.outflow),
    }
    values = {}
    fault = truth.fault
    for name, series in clean.items():
        noisy = series + rng.normal(0.0, sigma[name], size=series.size)
        if fault.kind in ("sensor_bias", "common_drift") and name in fault.sensors:
            onset = min(fault.onset, noisy.size)
            noisy[onset:] += fault.magnitude
        values[name] = noisy
    return Readings(values, dict(sigma), n)


def reading_names(n_steps: int) -> tuple[tuple[str, int], ...]:
    """The declared reading order: both storages on the grid, then the interval-mean flows."""
    names: list[tuple[str, int]] = []
    for sensor in ("storage_1", "storage_2"):
        names.extend((sensor, k) for k in range(n_steps))
    for sensor in ("inflow_gauge", "middle_gauge", "outflow_gauge"):
        names.extend((sensor, k) for k in range(n_steps - 1))
    return tuple(names)


def constraint_rows(config: MuskingumConfig, *, with_routing: bool) -> np.ndarray:
    """The declared rows on one interval's seven states.

    States, in order: storage 1 at the interval's start and end, storage 2 likewise, then the
    three interval-mean flows. Continuity DIFFERENCES the storages; routing AVERAGES them.

        continuity 1   (S1_end - S1_start) - dt (I1 - O1) = 0
        continuity 2   (S2_end - S2_start) - dt (O1 - O2) = 0
        routing 1      (S1_start + S1_end)/2 - K[x I1 + (1 - x) O1] = 0
        routing 2      (S2_start + S2_end)/2 - K[x O1 + (1 - x) O2] = 0

    Built from the DECLARED config, which need not be the truth.
    """
    K, x, dt = config.K, config.x, config.dt
    rows = [[-1.0, 1.0, 0.0, 0.0, -dt, dt, 0.0],
            [0.0, 0.0, -1.0, 1.0, 0.0, -dt, dt]]
    if with_routing:
        rows += [[0.5, 0.5, 0.0, 0.0, -K * x, -K * (1.0 - x), 0.0],
                 [0.0, 0.0, 0.5, 0.5, 0.0, -K * x, -K * (1.0 - x)]]
    return np.array(rows)


def design_matrix(config: MuskingumConfig, n_steps: int, *, with_routing: bool) -> np.ndarray:
    """The whole record's residual operator M, so that r = M y with b = 0.

    Rows are the declared constraints at each of the n_steps - 1 intervals; columns are the
    readings in `reading_names` order. Adjacent intervals SHARE a storage reading, so
    Cov(r) = M R M^T is not block diagonal and the sharing is carried exactly rather than
    approximated away.
    """
    rows = constraint_rows(config, with_routing=with_routing)
    names = reading_names(n_steps)
    index = {name: position for position, name in enumerate(names)}
    intervals = n_steps - 1
    M = np.zeros((rows.shape[0] * intervals, len(names)))
    for k in range(intervals):
        columns = [index[("storage_1", k)], index[("storage_1", k + 1)],
                   index[("storage_2", k)], index[("storage_2", k + 1)],
                   index[("inflow_gauge", k)], index[("middle_gauge", k)],
                   index[("outflow_gauge", k)]]
        for j, row in enumerate(rows):
            M[k * rows.shape[0] + j, columns] = row
    return M


def fault_reading_direction(kind: str, sensors: tuple[str, ...], onset: int,
                            n_steps: int) -> np.ndarray:
    """A candidate fault's effect on the READINGS, as a unit-magnitude profile.

    A persistent sensor bias adds one unit to that sensor from its onset onward. A physical
    lateral inflow is NOT a reading perturbation, so it is declared through the reading its
    absence corrupts -- the inflow gauge, which does not see it -- which is exactly why this
    testbed cannot separate the two.
    """
    names = reading_names(n_steps)
    direction = np.zeros(len(names))
    for position, (sensor, step) in enumerate(names):
        if sensor in sensors and step >= onset:
            direction[position] = 1.0
    if not direction.any():
        raise ValueError(f"fault {kind!r} on {sensors} at onset {onset} touches no reading")
    return direction
