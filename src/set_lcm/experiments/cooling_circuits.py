"""How many metered cooling circuits are worth buying, computed before any are.

`results/second_balance` measured a single-circuit cooling loop at **2 of 7** isolable faults
against the two-reach river's 5 of 7, and named the reason the loop is the worse design: what
it could not separate was INSTRUMENTAL rather than physical. It could not say which instrument
moved -- the question a shop would buy the tool to answer.

That study left one lever untested. A mould-cooling loop is not one circuit; it is a manifold
feeding several, and each circuit that carries its own flow meter and return thermocouple adds
rows as well as faults. Whether that trade is favourable is not obvious -- rows grow like 2N
and the catalogue like 3N -- and it is a procurement question, because metering a circuit costs
money and the answer is available before any is spent.

THE ANSWER IS NOT THE COUNT. Metering the whole manifold makes every declared fault
structurally isolable, and reading that as the result would be the mistake `fdi.Isolability`
warns about in its own docstring: structural separation says a residual direction exists, not
that a fault of realistic size would move it far enough to name. What binds this design is the
tightest separated pair, and this study reports the count and that pair together.

WHAT IS DECLARED. Per circuit: a flow meter and a return thermocouple. Shared: one supply
thermocouple at the header, and optionally a header flow meter. The supply temperature is NOT
a state and NOT a fault. It is a COEFFICIENT -- the enthalpy entering circuit i is the supply
temperature times circuit i's mass -- so an error in it is uncertainty in A, which is what
`ConstraintSet.A_var` exists to declare, and one shared thermocouple moves every energy row
together, which is why it is declared as a full vec(A) covariance rather than per row.

THE TEMPERATURE DATUM IS THE SUPPLY, and that is a correctness requirement rather than a
convenience. Enthalpy is measured from an arbitrary datum, so no reported geometry may depend
on where it is put. An earlier draft placed the datum below the supply and carried the offset
as the energy row's mass coefficient; the tightest pair then moved from 1.26x to 1.74x as the
offset ran from 0 to 20, which is a datum leaking into an answer. Writing the energy row on
the RISE removes the offset from A entirely. The supply thermocouple's uncertainty survives as
the variance of that now-zero coefficient, which is exactly what it is.

NONDIMENSIONAL THROUGHOUT. Mass is carried in units of one interval's nominal circuit
throughput, temperature in units of the mould-to-supply span, and energy in their product, so
every state is order one. The single-circuit loop's states were kilograms and joules, spanning
seven orders of magnitude, and a prior declared honestly in them was REFUSED by the kernel's
positive-definiteness tolerance at two of three swept scales. That refusal is what this
nondimensionalisation answers, and whether it is answered is a claim checked below.

WHAT THE DUTY ROW MUST USE. Effectiveness-NTU, Q = eps c_min (T_mould - T_supply), gives heat
proportional to circuit mass. `second_balance` found this is what separates the two temperature
sensors: a log-mean or arithmetic-mean form weights inlet and outlet equally and separates
neither. That finding is carried here rather than rediscovered.

    python -m set_lcm.experiments.cooling_circuits -> results/cooling_circuits.{md,json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import REPO_ROOT, header_line, provenance
from .second_balance import PRIOR_SWEEP, Topology, evaluate

CIRCUIT_SWEEP = (1, 2, 3, 4, 6)

# Declared operating point, nondimensional. Temperature is measured from the SUPPLY in units
# of the mould-to-supply span, so the supply sits at 0 and the mould face at 1. With
# effectiveness EPS a circuit returns at EPS and carries EPS of heat per unit mass, which is
# the duty coefficient.
EPS = 0.7
EPS_SWEEP = (0.2, 0.4, 0.7, 0.9)

# Declared uncertainty on the supply temperature, as a fraction of the mould-to-supply span.
# It is the variance of the energy row's mass coefficient, whose mean is zero by the choice of
# datum: the supply is where the datum is, to within this.
SUPPLY_THETA_SD = 0.05
SUPPLY_SD_SWEEP = (0.01, 0.05, 0.2, 0.5, 1.0)

REFERENCE_SD = 0.30      # prior sd on a cumulative state, in the units above

VARIANTS = ("energy only", "energy + duty", "energy + duty + header")
FULL = VARIANTS[-1]


def _slot(i: int, which: int) -> int:
    """State index: 0 the circuit's mass, 1 its returned enthalpy, 2 its heat."""
    return 1 + 3 * i + which


def topology(n: int, *, eps: float = EPS, supply_sd: float = SUPPLY_THETA_SD) -> Topology:
    """A manifold feeding `n` circuits, each metered for flow and return temperature.

    States: the header's cumulative mass, then per circuit its cumulative mass, the cumulative
    enthalpy it returns above supply, and the cumulative heat it absorbed.

    Rows, per circuit: an energy balance (heat absorbed equals the enthalpy rise it carries
    away) and the duty relation (heat proportional to mass at the declared effectiveness).
    Plus one header row, when the header is metered, that the circuits must sum to.

    Faults, per circuit. A FLOW-METER BIAS moves two states, not one: the same meter reading
    is what computes the enthalpy the circuit carries, so a bias of one mass unit also biases
    the returned enthalpy by the return temperature. Getting that coupling wrong is what makes
    a flow meter look invisible to an energy balance, and it is the reason the energy row sees
    a flow fault at all. A RETURN-THERMOCOUPLE BIAS moves the returned enthalpy alone, by the
    nominal mass. FOULING is not an instrument fault: the circuit transfers less heat and
    returns correspondingly less enthalpy, so it satisfies the energy balance exactly and
    violates only the duty relation. Plus the header meter's own bias.
    """
    if n < 1:
        raise ValueError("a cooling manifold has at least one circuit")
    states = ["header cumulative mass"]
    for i in range(n):
        states += [f"circuit {i + 1} cumulative mass",
                   f"circuit {i + 1} cumulative returned enthalpy",
                   f"circuit {i + 1} cumulative heat"]
    width = len(states)

    energy, duty = [], []
    for i in range(n):
        row = np.zeros(width)
        row[_slot(i, 1)], row[_slot(i, 2)] = -1.0, 1.0
        energy.append(row)
        row = np.zeros(width)
        row[_slot(i, 0)], row[_slot(i, 2)] = -eps, 1.0
        duty.append(row)
    header = np.zeros(width)
    header[0] = 1.0
    for i in range(n):
        header[_slot(i, 0)] = -1.0

    faults: dict[str, list[float]] = {}
    for i in range(n):
        f = np.zeros(width)
        f[_slot(i, 0)], f[_slot(i, 1)] = 1.0, eps       # the meter also computes the enthalpy
        faults[f"circuit {i + 1} flow-meter bias"] = f.tolist()
        f = np.zeros(width)
        f[_slot(i, 1)] = 1.0
        faults[f"circuit {i + 1} return-thermocouple bias"] = f.tolist()
        f = np.zeros(width)
        f[_slot(i, 1)], f[_slot(i, 2)] = 1.0, 1.0       # conserves energy; violates duty alone
        faults[f"circuit {i + 1} fouling"] = f.tolist()
    f = np.zeros(width)
    f[0] = 1.0
    faults["header flow-meter bias"] = f.tolist()

    variants = {
        "energy only": (np.array(energy), "one energy balance per circuit and nothing else"),
        "energy + duty": (np.array(energy + duty),
                          f"effectiveness-NTU duty per circuit at eps = {eps:g}"),
        FULL: (np.array(energy + duty + [header]),
               "the same, plus a metered header the circuits must sum to"),
    }
    return Topology(
        key=f"cooling_manifold_{n}",
        label=f"Mould-cooling manifold, {n} metered circuit{'' if n == 1 else 's'}",
        states=tuple(states),
        faults=faults,
        variants=variants,
        # The swept axis is how well the absorbed heat is known relative to metered
        # throughput, which is the thing a shop either has a credible number for or does not.
        stored_states=tuple(_slot(i, 2) for i in range(n)),
        reference_prior_sd=tuple([REFERENCE_SD] * width),
        prior_units="interval throughput for mass; interval throughput times the "
                    "mould-to-supply span for enthalpy and heat (nondimensional)",
        note=f"{n} circuits; the supply temperature is a coefficient, not a state",
        a_var={name: supply_theta_cov(n, int(rows.shape[0]), supply_sd)
               for name, (rows, _) in variants.items()},
        operating_point=tuple([float(n)] + [1.0, eps, eps] * n),
    )


def supply_theta_cov(n: int, rows: int, sd: float = SUPPLY_THETA_SD) -> np.ndarray:
    """Cov(vec(A)) for one shared supply thermocouple, in the row order the variants declare.

    The energy row's mass coefficient is the supply temperature, zero by the choice of datum
    and uncertain by `sd`. One instrument sets it for every circuit, so ONE error moves every
    energy row together: the rows are perfectly dependent, and the per-row (rows, n, n) form
    the schema also accepts cannot say that. This is the full vec(A) form, which can.
    """
    width = 1 + 3 * n
    cov = np.zeros((rows * width, rows * width))
    for i in range(n):                       # energy rows come first, one per circuit
        for j in range(n):
            cov[i * width + _slot(i, 0), j * width + _slot(j, 0)] = sd ** 2
    return cov


DATUM_SWEEP = (0.0, 1.0, 2.0, 5.0, 20.0)


def datum_offset_topology(n: int, offset: float, *, eps: float = EPS) -> Topology:
    """The REJECTED formulation, built only so the reason it was rejected can be computed.

    Here the temperature datum sits `offset` below the supply, so the energy row carries the
    offset as its mass coefficient and a flow-meter bias carries the offset in the enthalpy it
    miscomputes. Both are defensible-looking, and the residual A f is in fact datum-free -- the
    offset cancels between the enthalpy a biased meter adds and the enthalpy the row subtracts.
    What does NOT cancel is the row's own norm: A P A^T grows with the offset, so the whitening
    moves, and with it every reported angle. The datum is a bookkeeping choice, so a geometry
    that moves with it is reporting the bookkeeping. `topology` writes the row on the RISE and
    has no offset at all; this exists to show that the difference is not cosmetic, and offset
    0 reproduces it exactly.
    """
    top = topology(n, eps=eps)
    width = len(top.states)
    rows = {name: np.array(matrix, dtype=float) for name, (matrix, _) in top.variants.items()}
    for matrix in rows.values():
        for i in range(n):                                  # energy rows come first
            matrix[i, _slot(i, 0)] = offset
    faults = dict(top.faults)
    for i in range(n):
        f = np.zeros(width)
        f[_slot(i, 0)], f[_slot(i, 1)] = 1.0, offset + eps
        faults[f"circuit {i + 1} flow-meter bias"] = f.tolist()
    return Topology(
        key=f"cooling_manifold_{n}_datum_{offset:g}", label=top.label, states=top.states,
        faults=faults,
        variants={name: (rows[name], declares) for name, (_, declares) in top.variants.items()},
        stored_states=top.stored_states, reference_prior_sd=top.reference_prior_sd,
        prior_units=top.prior_units, note=f"rejected formulation, datum {offset:g} below supply",
        a_var=top.a_var, operating_point=tuple([float(n)] + [1.0, offset + eps, eps] * n))


def _circuit_of(name: str) -> str | None:
    """"circuit 3 fouling" -> "3"; the header meter belongs to no circuit."""
    parts = name.split()
    return parts[1] if parts[:1] == ["circuit"] else None


def _cell(top: Topology, variant: str, scale: float, *, declare_A_var: bool = False) -> dict:
    cell = evaluate(top, variant, scale, declare_A_var=declare_A_var)
    tight = cell["tightest_separated_pair"]
    return {
        "variant": variant, "prior_scale": scale,
        "A_declared_uncertain": declare_A_var,
        "refused": cell["refused"], "refusal": cell["refusal"],
        "declared_rows": cell["declared_rows"], "n_faults": cell["n_faults"],
        "residual_rank": cell["residual_rank"],
        "invisible": cell["invisible"], "isolable": cell["isolable"],
        "n_isolable": cell["n_isolable"],
        "confounded_pairs": cell["confounded_pairs"],
        "tightest_separated_pair": tight,
        "tightest_pair_is_within_one_circuit": (
            None if tight is None else
            _circuit_of(tight["a"]) is not None and _circuit_of(tight["a"]) == _circuit_of(tight["b"])),
    }


def _amplification(cell: dict) -> float | None:
    tight = cell["tightest_separated_pair"]
    return None if tight is None else float(tight["isolation_amplification"])


def compute() -> dict:
    sweep = []
    for n in CIRCUIT_SWEEP:
        top = topology(n)
        cells = [_cell(top, v, s) for v in VARIANTS for s in PRIOR_SWEEP]
        uncertain = [_cell(top, v, 1.0, declare_A_var=True) for v in VARIANTS]
        by_variant = {}
        for v in VARIANTS:
            mine = [c for c in cells if c["variant"] == v]
            reference = next(c for c in mine if c["prior_scale"] == 1.0)
            declared = next(c for c in uncertain if c["variant"] == v)
            by_variant[v] = {
                "declared_rows": reference["declared_rows"],
                "residual_rank": reference["residual_rank"],
                "refused": [c["prior_scale"] for c in mine if c["refused"]],
                "invisible": reference["invisible"],
                "isolable": reference["isolable"],
                "n_isolable": reference["n_isolable"],
                "confounded_pairs": reference["confounded_pairs"],
                "tightest_separated_pair": reference["tightest_separated_pair"],
                "tightest_pair_is_within_one_circuit": reference["tightest_pair_is_within_one_circuit"],
                "amplification_at_reference_prior": _amplification(reference),
                "amplification_by_prior": [
                    {"prior_scale": c["prior_scale"], "isolation_amplification": _amplification(c)}
                    for c in mine],
                "with_supply_temperature_declared_uncertain": {
                    "isolable": declared["isolable"],
                    "n_isolable": declared["n_isolable"],
                    "isolation_amplification": _amplification(declared),
                },
            }
        sweep.append({"circuits": n, "n_states": len(top.states), "n_faults": len(top.faults),
                      "by_variant": by_variant})

    effectiveness = []
    for eps in EPS_SWEEP:
        top = topology(4, eps=eps)
        effectiveness.append({"effectiveness": eps, "circuits": 4, **{
            v: {"n_isolable": (c := _cell(top, v, 1.0))["n_isolable"],
                "isolation_amplification": _amplification(c)} for v in VARIANTS}})
    supply = []
    for sd in SUPPLY_SD_SWEEP:
        top = topology(4, supply_sd=sd)
        supply.append({"supply_theta_sd": sd, "circuits": 4, **{
            v: {"n_isolable": (c := _cell(top, v, 1.0, declare_A_var=True))["n_isolable"],
                "isolation_amplification": _amplification(c)} for v in VARIANTS}})

    datum = [{"datum_below_supply": offset, "circuits": 4,
              "isolation_amplification":
                  _amplification(_cell(datum_offset_topology(4, offset), FULL, 1.0))}
             for offset in DATUM_SWEEP]

    out = {
        "schema_version": "fsre-cooling-circuits-v1",
        "scope": "A design study of a mould-cooling manifold, computed before any circuit is "
                 "metered. No record appears and nothing here describes an installed instrument.",
        "provenance": {"generation": provenance()},
        "declared": {
            "circuits_swept": list(CIRCUIT_SWEEP),
            "prior_scales_swept": list(PRIOR_SWEEP),
            "effectiveness": EPS,
            "effectiveness_swept": list(EPS_SWEEP),
            "supply_theta_sd": SUPPLY_THETA_SD,
            "supply_theta_sd_swept": list(SUPPLY_SD_SWEEP),
            "reference_prior_sd": REFERENCE_SD,
            "temperature_datum": "the supply header; the energy row is written on the rise, so "
                                 "no reported number depends on where enthalpy is zeroed",
            "units": "nondimensional: mass in interval throughput, temperature in the "
                     "mould-to-supply span, energy in their product",
            "why_nondimensional":
                "the single-circuit loop in results/second_balance declared its states in "
                "kilograms and joules, which span seven orders of magnitude, and the kernel's "
                "positive-definiteness tolerance refused a prior declared honestly in them at "
                "two of three swept scales. Nondimensionalising is the fix that study named.",
        },
        "sweep": sweep,
        "sensitivity": {
            "effectiveness": effectiveness,
            "supply_temperature_uncertainty": supply,
            "rejected_datum_formulation": datum,
            "note": "All three are run at four circuits and the reference prior. Structural "
                    "isolability does not move with any of them; the tightest pair does. The "
                    "datum rows are the REJECTED formulation, computed so the reason it was "
                    "rejected is a number rather than an assertion; the committed rows carry "
                    "no datum, and offset 0 reproduces them exactly.",
        },
        "limitations": [
            "A design study. Nothing here is measured on an installed manifold, and no fault is asserted to occur.",
            "Isolability is a property of the declared catalogue and the declared rows. A fault absent from the catalogue cannot be found, and the confounds reported are confounds of THIS catalogue.",
            "Structural isolability is not a detection guarantee. Every count in this report should be read with the tightest-pair amplification beside it, which is why none appears without one.",
            "Single-fault, static directions. Two circuits degrading together, and any temporal signature, are outside what this compares.",
            "The duty coefficient is one declared operating point. The effectiveness is swept rather than asserted, and the supply temperature's uncertainty is declared as A_var rather than assumed away, but the FORM of the duty relation is not varied.",
            "Fouling is declared as a process change that conserves energy. That it is a different KIND of thing from an instrument bias is an assumption of the catalogue, not a finding.",
            "The prior is declared, not measured. The heat-load axis it sweeps is the one the report finds binding, so a shop's real number for it matters more than anything else here.",
        ],
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition."""
    sweep, sens = r["sweep"], r["sensitivity"]

    def cells(variant):
        return [e["by_variant"][variant] for e in sweep]

    def amp(variant):
        return [c["amplification_at_reference_prior"] for c in cells(variant)]

    full, duty, bare = cells(FULL), cells("energy + duty"), cells("energy only")
    amp_full, amp_duty = amp(FULL), amp("energy + duty")
    prior_amps = [p["isolation_amplification"] for e in sweep
                  for p in e["by_variant"][FULL]["amplification_by_prior"]
                  if p["isolation_amplification"] is not None]
    eps_amps = [row[FULL]["isolation_amplification"] for row in sens["effectiveness"]]
    shift = [abs(c["with_supply_temperature_declared_uncertain"]["isolation_amplification"] - a) / a
             for variant in VARIANTS
             for c, a in zip(cells(variant), amp(variant))
             if a is not None
             and c["with_supply_temperature_declared_uncertain"]["isolation_amplification"] is not None]
    return {
        "nothing_is_refused_at_any_circuit_count_or_declared_prior": all(
            not c["refused"] for e in sweep for c in e["by_variant"].values()),
        "conservation_alone_cannot_see_a_fouling_circuit": all(
            all(f"circuit {i + 1} fouling" in c["invisible"] for i in range(e["circuits"]))
            for e, c in zip(sweep, bare)),
        "conservation_alone_isolates_nothing_at_any_circuit_count": all(
            c["n_isolable"] == 0 for c in bare),
        "the_duty_row_is_what_buys_isolation": all(
            d["n_isolable"] == 3 * e["circuits"] and d["n_isolable"] > b["n_isolable"]
            for e, d, b in zip(sweep, duty, bare)),
        "only_a_metered_header_makes_the_header_meter_visible": all(
            "header flow-meter bias" in b["invisible"] and "header flow-meter bias" in d["invisible"]
            and "header flow-meter bias" not in f["invisible"]
            for b, d, f in zip(bare, duty, full)),
        "the_full_arrangement_leaves_no_fault_structurally_confounded": all(
            c["n_isolable"] == e["n_faults"] and not c["confounded_pairs"]
            for e, c in zip(sweep, full)),
        "and_the_count_is_therefore_not_the_answer": max(amp_full) > 1.0,
        "without_a_header_the_hardest_pair_is_a_circuit_against_its_own_fouling": all(
            c["tightest_pair_is_within_one_circuit"]
            and c["tightest_separated_pair"]["b"].endswith("fouling") for c in duty),
        "metering_more_circuits_does_not_loosen_it_at_all": (
            max(amp_duty) - min(amp_duty) < 1e-9),
        "a_metered_header_starts_bound_by_the_header_meter_itself": (
            not full[0]["tightest_pair_is_within_one_circuit"]
            and "header flow-meter bias" in
            (full[0]["tightest_separated_pair"]["a"], full[0]["tightest_separated_pair"]["b"])),
        "and_that_pair_loosens_until_the_within_circuit_one_binds_instead": (
            full[-1]["tightest_pair_is_within_one_circuit"]
            and full[-1]["tightest_separated_pair"]["b"].endswith("fouling")
            and all(a > b for a, b in zip(amp_full, amp_full[1:]))),
        "converging_onto_the_same_within_circuit_floor": (
            amp_full[-1] > amp_duty[-1] and (amp_full[-1] - amp_duty[-1]) / amp_duty[-1] < 0.05),
        "the_declared_heat_load_binds_harder_than_the_meter_count": (
            max(prior_amps) / max(amp_full) > 10.0),
        "a_less_effective_circuit_is_the_harder_one_to_diagnose": all(
            a > b for a, b in zip(eps_amps, eps_amps[1:])),
        "declaring_the_supply_thermocouple_uncertain_changes_no_isolable_set": all(
            c["with_supply_temperature_declared_uncertain"]["isolable"] == c["isolable"]
            for variant in VARIANTS for c in cells(variant)),
        "and_moves_the_hardest_pair_by_under_one_percent": max(shift) < 0.01,
        "the_rejected_datum_formulation_moves_with_an_arbitrary_datum": (
            max(row["isolation_amplification"] for row in sens["rejected_datum_formulation"])
            / min(row["isolation_amplification"] for row in sens["rejected_datum_formulation"])
            > 1.2),
        "and_the_committed_one_reproduces_it_at_datum_zero": (
            sens["rejected_datum_formulation"][0]["datum_below_supply"] == 0.0
            and abs(sens["rejected_datum_formulation"][0]["isolation_amplification"]
                    - next(e for e in sweep if e["circuits"] == 4)["by_variant"][FULL]
                    ["amplification_at_reference_prior"]) < 1e-9),
        "the_sweep_reaches_at_least_six_circuits": max(e["circuits"] for e in sweep) >= 6,
    }


def render(report: dict) -> str:
    failed = [name for name, held in report["claims"].items() if not held]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; "
                           f"revise render() before writing")
    L: list[str] = []
    A = L.append
    d, sweep, sens = report["declared"], report["sweep"], report["sensitivity"]

    def cell(entry, variant):
        return entry["by_variant"][variant]

    def amp(entry, variant):
        value = cell(entry, variant)["amplification_at_reference_prior"]
        return "—" if value is None else f"{value:.2f}x"

    A("# How many metered cooling circuits are worth buying")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A("`results/second_balance` measured a single-circuit cooling loop at **2 of 7** isolable "
      "faults, against the two-reach river's 5 of 7, and named the reason the loop is the "
      "worse design: what it could not separate was *instrumental* rather than physical. It "
      "could not say which instrument moved — the question a shop would buy the tool to answer.")
    A("")
    A("That study left one lever untested. A mould-cooling loop is a manifold, not a circuit, "
      "and every circuit metered for flow and return temperature adds rows as well as faults. "
      "Metering a circuit costs money, so the trade is worth computing before any is spent.")
    A("")

    A("## What each row buys")
    A("")
    A("| circuits | faults | rows (full) | isolable: energy only | + duty | + header | hardest pair |")
    A("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for e in sweep:
        A(f"| {e['circuits']} | {e['n_faults']} | {cell(e, FULL)['declared_rows']} | "
          f"{cell(e, 'energy only')['n_isolable']} | "
          f"{cell(e, 'energy + duty')['n_isolable']} | "
          f"**{cell(e, FULL)['n_isolable']}** | {amp(e, FULL)} |")
    A("")
    A("**Conservation alone isolates nothing, at any number of circuits** — and the reason is "
      "worth stating, because it is the one a shop would not guess. A fouling circuit "
      "transfers less heat and returns correspondingly less enthalpy, so it satisfies the "
      "energy balance *exactly*: it is not merely hard to separate, it is **invisible**, in "
      "`null(A)`, at every size in this sweep. What conservation can see — a flow-meter bias "
      "and a return-thermocouple bias — it sees in one direction per circuit, so the two "
      "collapse into each other.")
    A("")
    A("**The duty row is what buys isolation.** It is the only row that a fouling circuit "
      "violates, and adding it takes the manifold from 0 isolable faults to "
      f"{cell(sweep[-1], 'energy + duty')['n_isolable']} of {sweep[-1]['n_faults']} at "
      f"{sweep[-1]['circuits']} circuits. The one it still cannot see is the header meter, "
      "which no per-circuit row touches; metering the header is what makes it visible, and "
      "that closes the catalogue.")
    A("")

    A("## Why the count is not the answer")
    A("")
    A("With every circuit and the header metered, **every declared fault is structurally "
      "isolable at every size in this sweep**. Read alone that is the wrong conclusion to "
      "draw, and `fdi.Isolability` says so in its own docstring: a structural separation "
      "means a residual direction exists, not that a fault of realistic size moves it far "
      "enough to name. The number that governs is the tightest separated pair.")
    A("")
    A("| circuits | hardest pair, + duty | hardest pair, + header | which pair |")
    A("| ---: | ---: | ---: | :--- |")
    for e in sweep:
        tight = cell(e, FULL)["tightest_separated_pair"]
        A(f"| {e['circuits']} | {amp(e, 'energy + duty')} | {amp(e, FULL)} | "
          f"`{tight['a']}` vs `{tight['b']}` |")
    A("")
    amp_duty = cell(sweep[-1], "energy + duty")["amplification_at_reference_prior"]
    first = cell(sweep[0], FULL)["amplification_at_reference_prior"]
    last = cell(sweep[-1], FULL)["amplification_at_reference_prior"]
    A(f"**Without a header meter, metering more circuits does not loosen the hardest pair at "
      f"all.** It reads {amp_duty:.2f}x at one circuit and {amp_duty:.2f}x at "
      f"{sweep[-1]['circuits']} — identical, not merely similar. The pair binding it is always "
      f"a circuit against *its own fouling*, a question answered entirely by that circuit's "
      f"two rows, so every circuit added is an independent copy of the same local problem "
      f"rather than evidence about any existing one. Extra circuits buy **coverage**, not "
      f"**conditioning**.")
    A("")
    A(f"**With a header meter the curve has two regimes**, and the table above shows the "
      f"handover. At small counts the binding pair is a circuit's flow meter against the "
      f"*header's* — with one or two circuits those are nearly the same measurement, and the "
      f"redundancy that makes the header meter worth having is also what makes the two hard "
      f"to tell apart. That pair loosens as circuits are added, from {first:.2f}x to "
      f"{last:.2f}x, until at {sweep[-1]['circuits']} circuits it is no longer the binding one: "
      f"the within-circuit thermocouple-against-fouling pair is, and the curve flattens onto "
      f"the {amp_duty:.2f}x floor the header meter cannot move. That floor, not the fault "
      f"count, is what this design is worth.")
    A("")

    A("## What actually binds")
    A("")
    scales = [row["prior_scale"] for row in cell(sweep[0], FULL)["amplification_by_prior"]]
    A("| prior on the heat load, relative to declared | "
      + " | ".join(f"{e['circuits']} circuit{'' if e['circuits'] == 1 else 's'}" for e in sweep)
      + " |")
    A("| ---: |" + " ---: |" * len(sweep))
    for scale in scales:
        row = [next(p["isolation_amplification"] for p in cell(e, FULL)["amplification_by_prior"]
                    if p["prior_scale"] == scale) for e in sweep]
        A(f"| {scale:g}x | " + " | ".join("—" if v is None else f"{v:.2f}x" for v in row) + " |")
    A("")
    worst = max(p["isolation_amplification"] for e in sweep
                for p in cell(e, FULL)["amplification_by_prior"]
                if p["isolation_amplification"] is not None)
    A(f"The swept axis is how well the absorbed heat is known relative to metered throughput, "
      f"and the whole circuit sweep is shown against it because that is the comparison that "
      f"matters. Loosening the declared heat load a hundredfold takes the hardest pair to "
      f"**{worst:.0f}x** — roughly two orders of magnitude worse than anything the circuit "
      f"count does, and it does that at every circuit count. A shop with a credible declared "
      f"heat load and two metered circuits is better placed than one with six and no idea what "
      f"the mould is absorbing. That is the procurement answer, and it is not the one the "
      f"count suggests.")
    A("")
    eps_rows = sens["effectiveness"]
    A(f"Effectiveness matters for the same reason: at eps = {eps_rows[0]['effectiveness']:g} the "
      f"hardest pair reads {eps_rows[0][FULL]['isolation_amplification']:.2f}x and at "
      f"{eps_rows[-1]['effectiveness']:g} it reads "
      f"{eps_rows[-1][FULL]['isolation_amplification']:.2f}x. A circuit that moves little heat "
      f"per unit of flow is the one this instrument diagnoses worst, which is unfortunate, "
      f"because it is also the one most likely to be fouling.")
    A("")

    A("## What the supply thermocouple costs")
    A("")
    A("One supply thermocouple sets a coefficient shared by every energy row, so a single "
      "error in it moves them all together. That is declared here as a full `vec(A)` "
      "covariance — the per-row form the schema also accepts cannot express the dependence — "
      f"at {d['supply_theta_sd']:g} of the mould-to-supply span, and carried through "
      "`Cov(E x)` into the residual covariance the geometry is computed from.")
    A("")
    A("| supply thermocouple sd | isolable, + header | hardest pair, + header |")
    A("| ---: | ---: | ---: |")
    for row in sens["supply_temperature_uncertainty"]:
        A(f"| {row['supply_theta_sd']:g} | {row[FULL]['n_isolable']} | "
          f"{row[FULL]['isolation_amplification']:.2f}x |")
    A("")
    A("**It is not the binding instrument.** Declaring it uncertain changes no isolable set "
      "anywhere in the sweep and moves the hardest pair by under 1% at the declared value. "
      "Even at a standard deviation equal to the entire mould-to-supply span — an instrument "
      "no one would install — the arrangement still isolates every declared fault. This is a "
      "negative result and it is the useful kind: it says where not to spend.")
    A("")
    A("The direction is not monotone, and that is a property of the quantity rather than "
      "noise. Amplification is a *ratio* of whitened lengths, so widening the residual "
      "covariance along one direction can rotate two signatures apart as easily as together, "
      "and the pair that binds can change identity as it does. The claim checked above is the "
      "one the numbers support: the isolable sets do not move, and the hardest pair barely does.")
    A("")

    A("## Units, and why they are not a detail here")
    A("")
    A(f"{d['why_nondimensional'][0].upper()}{d['why_nondimensional'][1:]}")
    A("")
    A("Every cell in this study is accepted at every swept prior. That is not a better result "
      "than the earlier study got; it is the same kernel being asked a question it can answer, "
      "and the difference is entirely a choice of units.")
    A("")
    A("## The temperature datum, and why the row is written on the rise")
    A("")
    datum = sens["rejected_datum_formulation"]
    A("Enthalpy is measured from an arbitrary zero, so no reported geometry may depend on "
      "where that zero is put. An earlier draft of this study carried the datum offset as the "
      "energy row's mass coefficient. The residual `A f` is genuinely datum-free — the offset "
      "cancels between the enthalpy a biased meter adds and the enthalpy the row subtracts — "
      "and that cancellation is exactly why the defect was easy to miss. What does not cancel "
      "is the row's own norm: `A P A^T` grows with the offset, so the whitening moves, and "
      "with it every angle reported from it.")
    A("")
    A("That formulation is still built here, so the reason it was rejected is a number:")
    A("")
    A("| datum below the supply | hardest pair, + header, 4 circuits |")
    A("| ---: | ---: |")
    for row in datum:
        A(f"| {row['datum_below_supply']:g} | {row['isolation_amplification']:.2f}x |")
    A("")
    A(f"A bookkeeping choice moves the answer by "
      f"{max(r['isolation_amplification'] for r in datum) / min(r['isolation_amplification'] for r in datum):.2f}x "
      f"over that range. Writing the row on the rise removes the offset from `A` entirely, so "
      f"there is no datum left to move — and at offset 0 the rejected formulation reproduces "
      f"the committed one exactly, which is what makes this a check rather than a comparison "
      f"of two different models. The supply thermocouple's uncertainty survives as the "
      f"variance of that now-zero coefficient, which is precisely what it is.")
    A("")

    A("## What this does not establish")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "cooling_circuits.md").write_text(text, encoding="utf-8")
    (out_dir / "cooling_circuits.json").write_text(
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
