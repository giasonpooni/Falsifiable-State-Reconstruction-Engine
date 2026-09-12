"""The public inputs of a run: what the estimator side is allowed to know besides
the observations, the declared prior and the declared constraint set.

    t            (N,) the step clock; report step k is at t[k], and observation j
                 is the one sampled at t[j]
    u_commanded  (N,) the commanded pump rate at each step -- a declared input,
                 not a measurement of what the pump did

A real record has no hidden truth, so runner.run() takes PublicInputs and nothing
that could carry one. A simulated run builds them with PublicInputs.from_truth(),
which copies exactly the two public fields and nothing else; the hidden arrays
the oracle bound needs are routed separately, by experiment code, through run()'s
explicit `oracle_inputs` keyword. The declared total a constraint author writes
down is not an input of the run: it reaches the estimator side only as the b of a
declared ConstraintSet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PublicInputs:
    t: np.ndarray            # (N,) step clock
    u_commanded: np.ndarray  # (N,) commanded pump rate

    def __post_init__(self):
        t = np.array(self.t, dtype=float, copy=True)
        u = np.array(self.u_commanded, dtype=float, copy=True)
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D; got shape {t.shape}")
        if u.shape != t.shape:
            raise ValueError(f"u_commanded has shape {u.shape}; the clock t has {t.shape}")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(u))):
            raise ValueError("public inputs must be finite")
        if t.size > 1 and not np.all(np.diff(t) > 0):
            raise ValueError("t must be strictly increasing")
        t.setflags(write=False)
        u.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "u_commanded", u)

    @classmethod
    def from_truth(cls, truth) -> "PublicInputs":
        """The public fields of a simulated truth, copied: its clock and its commanded
        input. Nothing hidden (the masses, the leak, the actual pump rate) is read."""
        return cls(t=truth.t, u_commanded=truth.u_commanded)
