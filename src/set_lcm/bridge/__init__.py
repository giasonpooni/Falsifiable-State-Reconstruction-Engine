"""Bridges from admitted evidence to the estimator side's Observation records.

    set_lcm.bridge.daf   serialized DAF (Data Acquisition Fabric) observation dicts
                         -> PublicInputs + list[Observation] + provenance, with refusals

A bridge consumes evidence another system admitted; it does not acquire, re-extract
or re-identify it, and it imports nothing from that system at runtime.
"""
