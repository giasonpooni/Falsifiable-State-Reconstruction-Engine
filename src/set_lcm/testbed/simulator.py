"""Two-reservoir material-transfer simulator (hidden ground truth).

    dm1/dt = -q(t)
    dm2/dt = +q(t) - leak(t)

Closed system  <=>  leak == 0  <=>  m1 + m2 = M for all t.

An unmodeled leak does not violate conservation of mass. It violates the
*assumption* that the two-reservoir boundary is closed. That is exactly the
distinction the constraint layer has to be able to detect rather than paper over.

Only `Truth.u_commanded` and `Truth.total0` are public to the estimator side
(they are declared model inputs). `Truth.m` and `Truth.leak` are hidden and
visible only to the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_VERSION = "reservoir2-linear-v1"


@dataclass(frozen=True)
class SimConfig:
    seed: int
    n_steps: int = 600
    dt: float = 1.0
    m0: tuple[float, float] = (70.0, 30.0)   # kg
    pump_commanded: float = 0.10             # kg/s, what the estimator is told
    pump_actual: float = 0.10                # kg/s, what actually happens (parameter error if != commanded)
    pump_window: tuple[int, int] = (50, 250)  # steps during which the pump runs
    pump_noise_std: float = 0.01             # kg/s, per-step fluctuation of the actual rate while pumping
    transfer_noise_std: float = 0.0          # kg/s, per-step unmodeled 1<->2 transfer at ALL steps (leaky valve)
    leak_rate: float = 0.0                   # kg/s lost from reservoir 2
    leak_window: tuple[int, int] = (300, 500)


@dataclass(frozen=True)
class Truth:
    t: np.ndarray            # (N,)
    m: np.ndarray            # (N, 2)  HIDDEN
    u_commanded: np.ndarray  # (N,)    public: commanded pump rate at each step
    leak: np.ndarray         # (N,)    HIDDEN
    total0: float            # declared initial total; what a constraint author would write down


def simulate(cfg: SimConfig) -> Truth:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    t = np.arange(n) * cfg.dt

    u_cmd = np.zeros(n)
    u_act = np.zeros(n)
    a, b = cfg.pump_window
    u_cmd[a:b] = cfg.pump_commanded
    u_act[a:b] = cfg.pump_actual + rng.normal(0.0, cfg.pump_noise_std, b - a)
    # drawn after the pump noise so scenarios with transfer_noise_std == 0 keep identical truth
    u_act = u_act + rng.normal(0.0, cfg.transfer_noise_std, n)

    leak = np.zeros(n)
    la, lb = cfg.leak_window
    leak[la:lb] = cfg.leak_rate

    m = np.zeros((n, 2))
    m[0] = cfg.m0
    for k in range(n - 1):
        q = u_act[k] * cfg.dt
        m[k + 1, 0] = m[k, 0] - q
        m[k + 1, 1] = m[k, 1] + q - leak[k] * cfg.dt

    return Truth(t=t, m=m, u_commanded=u_cmd, leak=leak, total0=float(sum(cfg.m0)))
