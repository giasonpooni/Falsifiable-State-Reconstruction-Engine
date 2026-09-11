"""Observation operator + degradation layer.

Maps hidden truth to what the estimator is allowed to see. Every degradation is
seed-controlled and declared in the Observation metadata where a real sensor
spec would declare it (noise and quantization go into R; missingness into the
mask; delay into arrival_t). Bias is *not* declared: it is a fault to detect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Observation
from .simulator import Truth

SOURCE_IDS = ("sensor_m1", "sensor_m2")


@dataclass(frozen=True)
class DegradeConfig:
    seed: int
    noise_std: tuple[float, float] = (2.0, 2.0)          # kg
    dropout_p: float = 0.0                                # per-sample, per-sensor Bernoulli loss
    blackout: tuple[int, int, int] | None = None          # (sensor, start_step, end_step)
    bias: tuple[int, float, int] | None = None            # (sensor, offset_kg, start_step) -- undeclared
    quant_step: float | None = None                       # kg
    delay_steps: int = 0                                  # uniform arrival delay


def observe(truth: Truth, cfg: DegradeConfig) -> list[Observation]:
    rng = np.random.default_rng(cfg.seed)
    n = len(truth.t)
    dt = float(truth.t[1] - truth.t[0]) if n > 1 else 1.0
    sig = np.asarray(cfg.noise_std, dtype=float)

    y = truth.m + rng.normal(0.0, 1.0, (n, 2)) * sig
    mask = ~(rng.random((n, 2)) < cfg.dropout_p)

    if cfg.bias is not None:
        s, off, k0 = cfg.bias
        y[k0:, s] += off

    R = np.diag(sig ** 2)
    if cfg.quant_step is not None:
        q = cfg.quant_step
        y = np.round(y / q) * q
        R = R + np.eye(2) * (q ** 2 / 12.0)   # declared uniform quantization variance

    if cfg.blackout is not None:
        s, k0, k1 = cfg.blackout
        mask[k0:k1, s] = False

    y = np.where(mask, y, np.nan)

    return [
        Observation(
            t=float(truth.t[k]),
            arrival_t=float(truth.t[k] + cfg.delay_steps * dt),
            y=y[k].copy(),
            R=R.copy(),
            mask=mask[k].copy(),
            source_ids=SOURCE_IDS,
        )
        for k in range(n)
    ]
