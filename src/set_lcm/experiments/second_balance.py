"""A design study: which second balance would actually let this repository isolate a fault.

Every constraint in this repository has rank(A) = 1. `fdi` proves what that costs: a scalar
residual means every visible fault direction is collinear with every other, so no fault can be
isolated from any other at any covariance, with any filter, given any amount of data. Detection
works today; isolation is blocked by algebra, not by effort. The roadmap's second independent
balance is therefore not one item among several -- it is the only thing that unblocks the
stated target of naming a degrading instrument.

Two candidates are on the roadmap: a two-reach Muskingum river design, and a cooling loop
carrying mass and energy balances over the same pipes. This study runs `fdi.isolability()` on
both BEFORE either is built, because a second balance does not automatically buy isolation: it
buys exactly the directions its rows separate, and the rest stay confounded.

NOT A MEASUREMENT. No instrument, no record and no estimate appears here. Every number is a
property of a PROPOSED constraint matrix, a DECLARED fault catalogue and a DECLARED prior. A
topology that isolates a fault here is one where the evidence could in principle name it; the
study says nothing about whether a real record would, at what magnitude, or how often.

What it computes, per topology and variant:

    STRUCTURAL, and independent of any covariance -- rank(A), which declared faults are
    invisible (in null(A)), which are pairwise collinear, and which are separated from every
    other visible candidate. This is the half that a better sensor cannot change.

    CONDITIONAL on the declared prior -- the whitened separation of the tightest pair and its
    isolation amplification, roughly how much larger an amplitude must be to isolate a fault
    than to detect one. This is the half that says whether a structural separation is worth
    anything in practice, and it is swept rather than declared once.

Each candidate is run in two variants: CONSERVATION ONLY, and conservation plus the model's
own CONSTITUTIVE relation (Muskingum routing; heat-exchanger duty). That contrast is the
study's main result, and it carries a cost the report states: every constitutive row's
coefficients are DECLARED PARAMETERS -- K and x, effectiveness and minimum heat-capacity rate
-- so the isolation it buys is conditional on parameters the kernel cannot currently carry
uncertainty for. That is the errors-in-variables gap, priced.

    python -m set_lcm.experiments.second_balance  -> results/second_balance.{md,json}
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..fdi import isolability
from ..schema import ConstraintSet
from .provenance import REPO_ROOT, header_line, provenance

# Declared Muskingum parameters. K is the storage-time constant in days and x the weighting
# between inflow and outflow; 0 <= x <= 0.5 is the standard range. Declared, never fitted here.
MUSKINGUM_K, MUSKINGUM_X = 1.0, 0.2
MUSKINGUM_SWEEP = tuple((k, x) for k in (0.5, 1.0, 2.0) for x in (0.0, 0.2, 0.35, 0.5))

# Declared cooling-loop operating point and heat-exchanger parameters.
CP_J_PER_KG_K, INTERVAL_S = 4186.0, 60.0
MDOT_KG_PER_S, T_IN_C, T_OUT_C = 2.0, 35.0, 25.0
EFFECTIVENESS, C_MIN_W_PER_K = 0.7, 8000.0
EFFECTIVENESS_SWEEP = (0.3, 0.5, 0.7, 0.9)

# The prior is swept, not declared once: structural isolability does not depend on it, but the
# amplification does. Each topology declares a REFERENCE PRIOR IN ITS OWN PHYSICAL UNITS and the
# sweep scales the stored-quantity states around it -- how well the stored quantity is known
# relative to throughput, which is the axis that means something.
#
# A dimensionless prior is not a neutral choice here and an earlier draft of this study used
# one. The cooling loop's states are kilograms and joules; giving both unit variance whitens
# joules against kilograms and reported the loop's tightest pair at 204,258x when its physical
# prior gives 2x. The conclusion that followed from that number was wrong. Declare the units.
PRIOR_SWEEP = (0.01, 1.0, 100.0)


@dataclass(frozen=True)
class Topology:
    key: str
    label: str
    states: tuple[str, ...]
    faults: dict[str, list[float]]
    variants: dict[str, tuple[np.ndarray, str]]   # name -> (rows, what the extra rows declare)
    stored_states: tuple[int, ...]                # indices the prior sweep scales
    reference_prior_sd: tuple[float, ...]         # declared, in this topology's own units
    prior_units: str
    note: str


def _muskingum(k: float = MUSKINGUM_K, x: float = MUSKINGUM_X) -> Topology:
    """Two reaches in series, on cumulative volumes over one interval.

    States are stored volume in each reach and cumulative volume past each of the three flow
    gauges, so continuity is exact arithmetic on the declared states. The routing rows are
    Muskingum's own constitutive relation S = K[x I + (1 - x) O] over the same interval, which
    is what makes storage and flow enter with DIFFERENT coefficients -- the whole reason it
    separates anything.
    """
    continuity = np.array([[1.0, 0.0, -1.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, -1.0, 1.0]])
    routing = np.array([[1.0, 0.0, -k * x, -k * (1.0 - x), 0.0],
                        [0.0, 1.0, 0.0, -k * x, -k * (1.0 - x)]])
    return Topology(
        key="muskingum_two_reach",
        label="Two-reach Muskingum river",
        states=("stored volume, reach 1", "stored volume, reach 2",
                "cumulative volume past the inflow gauge",
                "cumulative volume past the middle gauge",
                "cumulative volume past the outflow gauge"),
        faults={
            "storage 1 bias": [1, 0, 0, 0, 0],
            "storage 2 bias": [0, 1, 0, 0, 0],
            "inflow gauge bias": [0, 0, 1, 0, 0],
            "middle gauge bias": [0, 0, 0, 1, 0],
            "outflow gauge bias": [0, 0, 0, 0, 1],
            "common drift of all three flow gauges": [0, 0, 1, 1, 1],
            # NOT an instrument fault: water entering reach 1 that no gauge sees.
            "ungauged lateral inflow to reach 1": [0, 0, 1, 0, 0],
        },
        variants={
            "continuity only": (continuity, "two storage closures and nothing else"),
            "continuity + declared routing":
                (np.vstack([continuity, routing]),
                 f"Muskingum routing at declared K = {k:g} day, x = {x:g}"),
        },
        stored_states=(0, 1),
        reference_prior_sd=(1.0, 1.0, 1.0, 1.0, 1.0),
        prior_units="volume units, the same for every state",
        note="Cumulative volumes over one interval; continuity is exact on these states.",
    )


def _cooling_loop(effectiveness: float = EFFECTIVENESS) -> Topology:
    """Mass and energy over the same pipes, with the heat exchanger's own duty relation.

    States are stored mass and energy, cumulative mass and enthalpy past each port, and
    cumulative heat. Both conservation rows are exact +/-1 arithmetic on those states: the
    measured quantities do NOT enter A. They enter the FAULT DIRECTIONS instead, because a
    flow-meter bias corrupts both cumulative mass and cumulative enthalpy (weighted by the
    port temperature) while a temperature-sensor bias corrupts enthalpy alone (weighted by the
    flow). So the isolation geometry here is conditional on a declared operating point, which
    is where errors-in-variables bites this candidate.

    The duty row is effectiveness-NTU, Q = eps c_min (T_in - T_coolant_in). It involves the
    INLET temperature and not the outlet, which is what lets it separate the two temperature
    sensors; a log-mean or arithmetic-mean form weights them equally and separates neither.
    """
    inlet_enthalpy_per_kelvin = CP_J_PER_KG_K * MDOT_KG_PER_S * INTERVAL_S
    mass = [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
    energy = [0.0, 1.0, 0.0, 0.0, -1.0, 1.0, -1.0]
    duty = [0.0, 0.0, 0.0, 0.0,
            -effectiveness * C_MIN_W_PER_K / (CP_J_PER_KG_K * MDOT_KG_PER_S), 0.0, 1.0]
    return Topology(
        key="cooling_loop_mass_energy",
        label="Cooling loop, mass and energy over the same pipes",
        states=("stored mass", "stored energy", "cumulative mass in", "cumulative mass out",
                "cumulative enthalpy in", "cumulative enthalpy out", "cumulative heat"),
        faults={
            "inlet flow-meter bias":
                [0, 0, INTERVAL_S, 0, CP_J_PER_KG_K * T_IN_C * INTERVAL_S, 0, 0],
            "outlet flow-meter bias":
                [0, 0, 0, INTERVAL_S, 0, CP_J_PER_KG_K * T_OUT_C * INTERVAL_S, 0],
            "inlet temperature bias": [0, 0, 0, 0, inlet_enthalpy_per_kelvin, 0, 0],
            "outlet temperature bias": [0, 0, 0, 0, 0, inlet_enthalpy_per_kelvin, 0],
            "stored-mass sensor bias": [1, 0, 0, 0, 0, 0, 0],
            "stored-energy sensor bias": [0, 1, 0, 0, 0, 0, 0],
            # NOT an instrument fault: the exchanger transferring other than declared.
            "heat-duty error": [0, 0, 0, 0, 0, 0, 1],
        },
        variants={
            "mass + energy only": (np.array([mass, energy]),
                                   "two conservation rows and nothing else"),
            "mass + energy + declared duty":
                (np.array([mass, energy, duty]),
                 f"effectiveness-NTU duty at declared eps = {effectiveness:g}, "
                 f"c_min = {C_MIN_W_PER_K:g} W/K"),
        },
        stored_states=(0, 1),
        # Declared in kilograms and joules, at the scale of one interval at the operating
        # point: 120 kg through each port, about 1.8e7 J of enthalpy, about 5e6 J of heat.
        reference_prior_sd=(1.0, 1e5, 2.0, 2.0, 2e5, 2e5, 1e5),
        prior_units="kg for mass states, J for energy states",
        note="Conservation rows are exact on these states; the operating point enters the "
             "fault directions, not A.",
    )


def _ridgway_today() -> Topology:
    """What the repository has now, for the row the comparison is against."""
    return Topology(
        key="ridgway_today",
        label="Ridgway closure (every constraint in this repository today)",
        states=("stored volume", "cumulative gauged net inflow"),
        faults={
            "storage gauge bias": [1, 0],
            "inflow gauge bias": [0, 1],
            "ungauged catchment inflow": [0, 1],
            "equal and opposite inflow/outflow bias": [0, 0],
        },
        variants={"one closure": (np.array([[1.0, -1.0]]), "the single storage closure")},
        stored_states=(0,),
        reference_prior_sd=(1.0, 1.0),
        prior_units="volume units, the same for both states",
        note="rank 1: a scalar residual, so no pair of visible faults can be separated.",
    )


TOPOLOGIES = (_ridgway_today(), _muskingum(), _cooling_loop())


def _prior(topology: Topology, scale: float) -> np.ndarray:
    """The topology's declared reference prior, with the stored states scaled by `scale`.

    In the topology's own physical units throughout. Whitening a joule against a kilogram is
    a choice of units masquerading as a choice of prior, and it changes the answer.
    """
    sd = np.array(topology.reference_prior_sd, dtype=float)
    sd[list(topology.stored_states)] *= scale
    return np.diag(sd ** 2)


def evaluate(topology: Topology, variant: str, scale: float) -> dict:
    """One topology, one variant, one declared prior -- or the kernel's refusal of it.

    A refusal is reported as a cell outcome rather than tuned around. The cooling loop's
    states are kilograms and joules, so a prior declared honestly in those units spans enough
    orders of magnitude that the kernel's positive-definiteness tolerance rejects it. That is
    a measured property of the design, and the argument for nondimensionalising it.
    """
    rows, declares = topology.variants[variant]
    cs = ConstraintSet(f"{topology.key}:{variant}", rows, np.zeros(rows.shape[0]),
                       f"{topology.label} -- {declares}")
    try:
        result = isolability(topology.faults, _prior(topology, scale), cs)
    except ValueError as refusal:
        return {"variant": variant, "declares": declares, "prior_scale": scale,
                "declared_rows": int(rows.shape[0]), "n_faults": len(topology.faults),
                "refused": True, "refusal": str(refusal).split(";")[0],
                "residual_rank": None, "visible": None, "invisible": None,
                "isolable": None, "n_isolable": None, "confounded_pairs": None,
                "tightest_separated_pair": None, "pairs": None}
    archived = result.as_dict()
    pairs = archived["pairs"]
    separated = [p for p in pairs if p["orthogonal_fraction"] is not None
                 and np.isfinite(p["orthogonal_fraction"]) and p["orthogonal_fraction"] > 0.0]
    tightest = min(separated, key=lambda p: p["orthogonal_fraction"]) if separated else None
    return {
        "variant": variant, "declares": declares, "prior_scale": scale,
        "declared_rows": int(rows.shape[0]), "residual_rank": archived["residual_rank"],
        "n_faults": len(topology.faults), "refused": False, "refusal": None,
        "visible": archived["visible"], "invisible": archived["invisible"],
        "isolable": archived["isolable"],
        "n_isolable": len(archived["isolable"]),
        "confounded_pairs": [[p["a"], p["b"]] for p in pairs
                             if p["orthogonal_fraction"] == 0.0],
        "tightest_separated_pair": (None if tightest is None else {
            "a": tightest["a"], "b": tightest["b"],
            "orthogonal_fraction": tightest["orthogonal_fraction"],
            "isolation_amplification": tightest["isolation_amplification"]}),
        "pairs": pairs,
    }


def parameter_sensitivity() -> dict:
    """Is the headline an artifact of one declared parameter value? Swept, not asserted."""
    muskingum = []
    for k, x in MUSKINGUM_SWEEP:
        topology = _muskingum(k, x)
        cell = evaluate(topology, "continuity + declared routing", 1.0)
        muskingum.append({"K_days": k, "x": x, "n_isolable": cell["n_isolable"],
                          "residual_rank": cell["residual_rank"],
                          "isolable": cell["isolable"]})
    cooling = []
    for effectiveness in EFFECTIVENESS_SWEEP:
        topology = _cooling_loop(effectiveness)
        cell = evaluate(topology, "mass + energy + declared duty", 1.0)
        cooling.append({"effectiveness": effectiveness, "n_isolable": cell["n_isolable"],
                        "residual_rank": cell["residual_rank"], "isolable": cell["isolable"]})
    return {
        "muskingum_routing": muskingum,
        "cooling_loop_duty": cooling,
        "note": "Structural isolability does not depend on the prior, so only the declared "
                "constitutive parameters are swept here; the prior sweep varies the "
                "amplification instead.",
    }


def compute() -> dict:
    topologies = []
    for topology in TOPOLOGIES:
        variants = []
        for variant in topology.variants:
            cells = [evaluate(topology, variant, scale) for scale in PRIOR_SWEEP]
            accepted = [c for c in cells if not c["refused"]]
            if not accepted:
                raise ValueError(f"{topology.key}:{variant} was refused at every declared prior")
            structural = accepted[0]
            amplifications = [c["tightest_separated_pair"]["isolation_amplification"]
                              for c in accepted if c["tightest_separated_pair"]]
            variants.append({
                "variant": variant, "declares": structural["declares"],
                "declared_rows": structural["declared_rows"],
                "residual_rank": structural["residual_rank"],
                "n_faults": structural["n_faults"],
                "visible": structural["visible"], "invisible": structural["invisible"],
                "isolable": structural["isolable"], "n_isolable": structural["n_isolable"],
                "confounded_pairs": structural["confounded_pairs"],
                "by_prior": [{"prior_scale": c["prior_scale"], "refused": c["refused"],
                              "refusal": c["refusal"], "isolable": c["isolable"],
                              "tightest_separated_pair": c["tightest_separated_pair"]}
                             for c in cells],
                "n_priors_refused": len(cells) - len(accepted),
                "amplification_range": ([min(amplifications), max(amplifications)]
                                        if amplifications else None),
            })
        topologies.append({"key": topology.key, "label": topology.label,
                           "states": list(topology.states), "note": topology.note,
                           "faults": {name: list(map(float, direction))
                                      for name, direction in topology.faults.items()},
                           "variants": variants})
    out = {
        "schema_version": "fsre-second-balance-study-v1",
        "scope": "A design study of PROPOSED constraint topologies. No instrument, record or "
                 "estimate appears here; nothing below is a measurement.",
        "provenance": {"generation": provenance()},
        "declared": {
            "muskingum": {"K_days": MUSKINGUM_K, "x": MUSKINGUM_X,
                          "status": "declared parameters of the routing relation; never fitted here",
                          "swept": [list(pair) for pair in MUSKINGUM_SWEEP]},
            "cooling_loop": {"cp_J_per_kg_K": CP_J_PER_KG_K, "interval_s": INTERVAL_S,
                             "mdot_kg_per_s": MDOT_KG_PER_S, "T_in_C": T_IN_C,
                             "T_out_C": T_OUT_C, "effectiveness": EFFECTIVENESS,
                             "c_min_W_per_K": C_MIN_W_PER_K,
                             "status": "declared operating point; it enters the fault "
                                       "directions rather than A, and is not measured here",
                             "effectiveness_swept": list(EFFECTIVENESS_SWEEP)},
            "prior_sweep": list(PRIOR_SWEEP),
            "prior_status": "unit variance on throughput states, the swept scale squared on "
                            "stored states; structural isolability does not depend on it",
        },
        "topologies": topologies,
        "parameter_sensitivity": parameter_sensitivity(),
        "limitations": [
            "No instrument, record or estimate appears in this study; nothing here is a measurement.",
            "Isolability is structural. A topology that separates two faults here says the evidence could in principle name one, not that a real record would, at what magnitude, or how often.",
            "Every fault carries one unrestricted signed amplitude, so a direction and its negation are the same line; a sign convention cannot separate a pair this study calls confounded.",
            "Each constitutive row's coefficients are declared parameters. The kernel carries uncertainty on b through b_var and none on A, so the isolation a constitutive row buys is conditional on parameters whose uncertainty cannot yet be represented.",
            "Simultaneous faults, temporal signatures and unknown onset are outside what fdi.isolability scores.",
            "The cooling loop's states span seven orders of magnitude in natural units; a prior declared directly in those units can fail the kernel's positive-definiteness tolerance, which is a practical argument for nondimensionalising that design before building it.",
        ],
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition."""
    by_key = {t["key"]: {v["variant"]: v for v in t["variants"]} for t in r["topologies"]}
    today = by_key["ridgway_today"]["one closure"]
    river_bare = by_key["muskingum_two_reach"]["continuity only"]
    river_full = by_key["muskingum_two_reach"]["continuity + declared routing"]
    loop_bare = by_key["cooling_loop_mass_energy"]["mass + energy only"]
    loop_full = by_key["cooling_loop_mass_energy"]["mass + energy + declared duty"]
    sensitivity = r["parameter_sensitivity"]
    return {
        "the_repository_today_isolates_nothing": today["n_isolable"] == 0,
        "the_repository_today_has_a_scalar_residual": today["residual_rank"] == 1,
        "conservation_alone_isolates_almost_nothing": (river_bare["n_isolable"] <= 1
                                                       and loop_bare["n_isolable"] <= 1),
        "a_constitutive_row_isolates_strictly_more_than_conservation_alone": (
            river_full["n_isolable"] > river_bare["n_isolable"]
            and loop_full["n_isolable"] > loop_bare["n_isolable"]),
        "the_river_design_isolates_more_than_the_cooling_loop":
            river_full["n_isolable"] > loop_full["n_isolable"],
        "the_rivers_remaining_confound_is_physical_not_instrumental": (
            [sorted(pair) for pair in river_full["confounded_pairs"]]
            == [sorted(["inflow gauge bias", "ungauged lateral inflow to reach 1"])]),
        "the_common_flow_drift_is_invisible_without_routing": (
            "common drift of all three flow gauges" in river_bare["invisible"]
            and "common drift of all three flow gauges" not in river_full["invisible"]),
        "structural_isolability_does_not_move_with_the_prior": all(
            cell["isolable"] == variant["isolable"]
            for topology in r["topologies"] for variant in topology["variants"]
            for cell in variant["by_prior"] if not cell["refused"]),
        "every_separation_this_study_reports_is_actionable": all(
            variant["amplification_range"] is None or variant["amplification_range"][1] < 100.0
            for topology in r["topologies"] for variant in topology["variants"]),
        "the_cooling_loops_declared_units_are_refused_at_some_prior_scales": all(
            variant["n_priors_refused"] > 0
            for variant in by_key["cooling_loop_mass_energy"].values()),
        "the_river_design_accepts_every_declared_prior": all(
            variant["n_priors_refused"] == 0
            for variant in by_key["muskingum_two_reach"].values()),
        "the_river_result_holds_across_the_declared_parameter_sweep": (
            min(cell["n_isolable"] for cell in sensitivity["muskingum_routing"])
            >= river_full["n_isolable"] - 1),
        "the_cooling_loop_result_holds_across_its_sweep": (
            len({cell["n_isolable"] for cell in sensitivity["cooling_loop_duty"]}) == 1),
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    by_key = {t["key"]: {v["variant"]: v for v in t["variants"]} for t in report["topologies"]}
    A("# Which second balance would let this repository isolate a fault")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A("Every constraint in this repository has rank(A) = 1, which means a scalar residual, "
      "which means every visible fault direction is collinear with every other. No fault can "
      "be isolated from any other at any covariance, with any filter, given any amount of "
      "data. This study asks what a second balance would change, before one is built.")
    A("")

    A("## The result")
    A("")
    A("| topology | rows | residual rank | faults isolable | structurally invisible | perfectly confounded pairs |")
    A("|---|---:|---:|---:|---:|---:|")
    for topology in report["topologies"]:
        for variant in topology["variants"]:
            name = (f"{topology['label']}" if len(topology["variants"]) == 1
                    else f"{topology['label']} — *{variant['variant']}*")
            A(f"| {name} | {variant['declared_rows']} | {variant['residual_rank']} | "
              f"**{variant['n_isolable']} of {variant['n_faults']}** | "
              f"{len(variant['invisible'])} | {len(variant['confounded_pairs'])} |")
    A("")

    river_bare = by_key["muskingum_two_reach"]["continuity only"]
    river_full = by_key["muskingum_two_reach"]["continuity + declared routing"]
    loop_bare = by_key["cooling_loop_mass_energy"]["mass + energy only"]
    loop_full = by_key["cooling_loop_mass_energy"]["mass + energy + declared duty"]

    A("**Conservation alone buys almost nothing.** A second *conservation* row takes the river "
      f"from 0 isolable faults to {river_bare['n_isolable']} and the cooling loop to "
      f"{loop_bare['n_isolable']}, out of {river_bare['n_faults']} declared candidates each. "
      "Adding a balance is not the same as adding information about which instrument moved.")
    A("")
    A("**The constitutive relation is what separates sensors.** Muskingum's routing law makes "
      "storage and flow enter with different coefficients, and the cooling loop's duty relation "
      "involves the inlet temperature and not the outlet. With those rows the river reaches "
      f"**{river_full['n_isolable']} of {river_full['n_faults']}** and the loop "
      f"**{loop_full['n_isolable']} of {loop_full['n_faults']}**.")
    A("")
    A("**The river design dominates, on two counts.** It isolates more than twice as many "
      "candidates, and what it still cannot separate is physical rather than instrumental:")
    A("")
    for pair in river_full["confounded_pairs"]:
        A(f"- `{pair[0]}` and `{pair[1]}` are the same direction. Water arriving that no gauge "
          "sees, and an inflow gauge reading high, are the same statement about the evidence. "
          "No topology fixes that; only a gauge on the lateral inflow does.")
    A("")
    A("The cooling loop's remaining confounds are instrumental, which is worse:")
    A("")
    for pair in loop_full["confounded_pairs"]:
        A(f"- `{pair[0]}` and `{pair[1]}` are the same direction.")
    A("")
    A("With conservation alone the loop leaves "
      f"{len(loop_bare['confounded_pairs'])} pairs perfectly confounded: one energy equation "
      "gives one residual direction, so every fault that touches only the energy side collapses "
      "into it, whatever the sensors are.")
    A("")

    A("## What routing makes visible")
    A("")
    A("A common drift of all three flow gauges is in the null space of continuity — the classic "
      "invisible fault, the river's version of an equal-and-opposite bias. The routing rows "
      f"weight the gauges differently, so it becomes visible and in fact isolable: it appears in "
      f"`{river_bare['variant']}`'s invisible list and not in `{river_full['variant']}`'s.")
    A("")

    A("## The price: every constitutive row is a declared parameter")
    A("")
    A("Routing rows carry K and x. The duty row carries effectiveness and the minimum "
      "heat-capacity rate. The kernel declares uncertainty on `b` through `b_var` and carries "
      "**none on `A`**, so the isolation a constitutive row buys is conditional on parameters "
      "whose uncertainty cannot currently be represented. Swept rather than assumed:")
    A("")
    A("| Muskingum K (days) | x | residual rank | isolable |")
    A("|---:|---:|---:|---:|")
    for cell in report["parameter_sensitivity"]["muskingum_routing"]:
        A(f"| {cell['K_days']:g} | {cell['x']:g} | {cell['residual_rank']} | "
          f"{cell['n_isolable']} of {river_full['n_faults']} |")
    A("")
    A("| heat-exchanger effectiveness | residual rank | isolable |")
    A("|---:|---:|---:|")
    for cell in report["parameter_sensitivity"]["cooling_loop_duty"]:
        A(f"| {cell['effectiveness']:g} | {cell['residual_rank']} | "
          f"{cell['n_isolable']} of {loop_full['n_faults']} |")
    A("")
    counts = {cell["n_isolable"] for cell in report["parameter_sensitivity"]["muskingum_routing"]}
    if len(counts) > 1:
        A(f"The river result is {max(counts)} of {river_full['n_faults']} across the declared "
          f"range and drops to {min(counts)} at the corner of it. A design study that reported "
          "only its chosen parameter pair would not show that.")
        A("")

    A("## Structural and conditional, kept apart")
    A("")
    A("Which faults are isolable is structural: it does not move when the declared prior does, "
      "and this study checks that across a prior sweep of "
      f"{report['declared']['prior_sweep']}. What does move is the amplification — how much "
      "larger an amplitude must be to isolate a fault than to detect one:")
    A("")
    A("| topology | variant | tightest separated pair | amplification across the prior sweep |")
    A("|---|---|---|---|")
    for topology in report["topologies"]:
        for variant in topology["variants"]:
            accepted = [c for c in variant["by_prior"] if not c["refused"]]
            tight = accepted[0]["tightest_separated_pair"] if accepted else None
            span = variant["amplification_range"]
            A(f"| {topology['label']} | {variant['variant']} | "
              f"{'none separated' if tight is None else f'{tight[chr(97)]} vs {tight[chr(98)]}'} | "
              f"{'—' if span is None else f'{span[0]:.2f}x to {span[1]:.2f}x'} |")
    A("")
    A("An amplification near 1 means a fault that can be detected can be named; a large one "
      "means the separation is real and useless. **Every separation in this study is "
      "actionable** -- the largest is "
      f"{max(v['amplification_range'][1] for t in report['topologies'] for v in t['variants'] if v['amplification_range']):.1f}x. "
      "An earlier draft of this study reported the cooling loop's tightest pair at 204,258x. "
      "That was a units error, not a finding: it whitened joules against kilograms by giving "
      "both unit variance. Declared in kilograms and joules the same pair reads "
      f"{by_key['cooling_loop_mass_energy']['mass + energy + declared duty']['amplification_range'][0]:.2f}x. "
      "A dimensionless prior is not a neutral choice.")
    A("")
    n_scales = len(report["declared"]["prior_sweep"])
    refused = [(t["label"], v["variant"], v["n_priors_refused"])
               for t in report["topologies"] for v in t["variants"] if v["n_priors_refused"]]
    if refused:
        worst = max(count for _, _, count in refused)
        A("**The cooling loop's own units are near the kernel's numerical limit.** Its states "
          "are kilograms and joules, which span enough orders of magnitude that a prior "
          "declared honestly in them is rejected by `check_spd`'s positive-definiteness "
          f"tolerance at {worst} of the {n_scales} swept scales, in "
          f"{'both variants' if len(refused) > 1 else 'one variant'}. Those cells are reported "
          "as refusals rather than tuned around: the refused scales are listed per variant in "
          "the JSON. The river's states are all volumes and every declared prior is accepted. "
          "Nondimensionalising the loop would fix this, and is work the river design does not "
          "need.")
        A("")

    A("## What this study does not show")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "second_balance.md").write_text(text, encoding="utf-8")
    (out_dir / "second_balance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
