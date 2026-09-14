"""Fluid State Reconstruction Testbed (FSRT).

Subpackages:
    set_lcm.schema       typed envelopes (Observation, ConstraintSet, StateEstimate, Status)
    set_lcm.lcm          constraint-reconciliation kernel and consistency statistic
    set_lcm.testbed      simulator, degradation layer, estimators, runner, evaluator
    set_lcm.experiments  scenario grids and report writers
    set_lcm.bridge       admitted evidence -> Observation records (bridge.daf: serialized DAF
                         NOAA observations, with refusals and provenance; no DAF import at runtime)
"""
