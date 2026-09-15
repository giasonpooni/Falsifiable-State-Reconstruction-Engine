# How many metered cooling circuits are worth buying

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v33-x86_64-with-glibc2.39; source sha256 1d22ad6e1b9a, git 385bafd04b. Latency columns are wall-clock on this machine and are not a claim.

A design study of a mould-cooling manifold, computed before any circuit is metered. No record appears and nothing here describes an installed instrument.

`results/second_balance` measured a single-circuit cooling loop at **2 of 7** isolable faults, against the two-reach river's 5 of 7, and named the reason the loop is the worse design: what it could not separate was *instrumental* rather than physical. It could not say which instrument moved — the question a shop would buy the tool to answer.

That study left one lever untested. A mould-cooling loop is a manifold, not a circuit, and every circuit metered for flow and return temperature adds rows as well as faults. Metering a circuit costs money, so the trade is worth computing before any is spent.

## What each row buys

| circuits | faults | rows (full) | isolable: energy only | + duty | + header | hardest pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 3 | 0 | 3 | **4** | 1.42x |
| 2 | 7 | 5 | 0 | 6 | **7** | 1.33x |
| 3 | 10 | 7 | 0 | 9 | **10** | 1.27x |
| 4 | 13 | 9 | 0 | 12 | **13** | 1.26x |
| 6 | 19 | 13 | 0 | 18 | **19** | 1.25x |

**Conservation alone isolates nothing, at any number of circuits** — and the reason is worth stating, because it is the one a shop would not guess. A fouling circuit transfers less heat and returns correspondingly less enthalpy, so it satisfies the energy balance *exactly*: it is not merely hard to separate, it is **invisible**, in `null(A)`, at every size in this sweep. What conservation can see — a flow-meter bias and a return-thermocouple bias — it sees in one direction per circuit, so the two collapse into each other.

**The duty row is what buys isolation.** It is the only row that a fouling circuit violates, and adding it takes the manifold from 0 isolable faults to 18 of 19 at 6 circuits. The one it still cannot see is the header meter, which no per-circuit row touches; metering the header is what makes it visible, and that closes the catalogue.

## Why the count is not the answer

With every circuit and the header metered, **every declared fault is structurally isolable at every size in this sweep**. Read alone that is the wrong conclusion to draw, and `fdi.Isolability` says so in its own docstring: a structural separation means a residual direction exists, not that a fault of realistic size moves it far enough to name. The number that governs is the tightest separated pair.

| circuits | hardest pair, + duty | hardest pair, + header | which pair |
| ---: | ---: | ---: | :--- |
| 1 | 1.23x | 1.42x | `circuit 1 flow-meter bias` vs `header flow-meter bias` |
| 2 | 1.23x | 1.33x | `circuit 1 flow-meter bias` vs `header flow-meter bias` |
| 3 | 1.23x | 1.27x | `circuit 2 flow-meter bias` vs `header flow-meter bias` |
| 4 | 1.23x | 1.26x | `circuit 3 return-thermocouple bias` vs `circuit 3 fouling` |
| 6 | 1.23x | 1.25x | `circuit 5 return-thermocouple bias` vs `circuit 5 fouling` |

**Without a header meter, metering more circuits does not loosen the hardest pair at all.** It reads 1.23x at one circuit and 1.23x at 6 — identical, not merely similar. The pair binding it is always a circuit against *its own fouling*, a question answered entirely by that circuit's two rows, so every circuit added is an independent copy of the same local problem rather than evidence about any existing one. Extra circuits buy **coverage**, not **conditioning**.

**With a header meter the curve has two regimes**, and the table above shows the handover. At small counts the binding pair is a circuit's flow meter against the *header's* — with one or two circuits those are nearly the same measurement, and the redundancy that makes the header meter worth having is also what makes the two hard to tell apart. That pair loosens as circuits are added, from 1.42x to 1.25x, until at 6 circuits it is no longer the binding one: the within-circuit thermocouple-against-fouling pair is, and the curve flattens onto the 1.23x floor the header meter cannot move. That floor, not the fault count, is what this design is worth.

## What actually binds

| prior on the heat load, relative to declared | 1 circuit | 2 circuits | 3 circuits | 4 circuits | 6 circuits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01x | 1.41x | 1.41x | 1.41x | 1.41x | 1.41x |
| 1x | 1.42x | 1.33x | 1.27x | 1.26x | 1.25x |
| 100x | 110.51x | 93.35x | 86.28x | 85.51x | 84.57x |

The swept axis is how well the absorbed heat is known relative to metered throughput, and the whole circuit sweep is shown against it because that is the comparison that matters. Loosening the declared heat load a hundredfold takes the hardest pair to **111x** — roughly two orders of magnitude worse than anything the circuit count does, and it does that at every circuit count. A shop with a credible declared heat load and two metered circuits is better placed than one with six and no idea what the mould is absorbing. That is the procurement answer, and it is not the one the count suggests.

Effectiveness matters for the same reason: at eps = 0.2 the hardest pair reads 2.47x and at 0.9 it reads 1.21x. A circuit that moves little heat per unit of flow is the one this instrument diagnoses worst, which is unfortunate, because it is also the one most likely to be fouling.

## What the supply thermocouple costs

One supply thermocouple sets a coefficient shared by every energy row, so a single error in it moves them all together. That is declared here as a full `vec(A)` covariance — the per-row form the schema also accepts cannot express the dependence — at 0.05 of the mould-to-supply span, and carried through `Cov(E x)` into the residual covariance the geometry is computed from.

| supply thermocouple sd | isolable, + header | hardest pair, + header |
| ---: | ---: | ---: |
| 0.01 | 13 | 1.26x |
| 0.05 | 13 | 1.25x |
| 0.2 | 13 | 1.19x |
| 0.5 | 13 | 1.17x |
| 1 | 13 | 1.17x |

**It is not the binding instrument.** Declaring it uncertain changes no isolable set anywhere in the sweep and moves the hardest pair by under 1% at the declared value. Even at a standard deviation equal to the entire mould-to-supply span — an instrument no one would install — the arrangement still isolates every declared fault. This is a negative result and it is the useful kind: it says where not to spend.

The direction is not monotone, and that is a property of the quantity rather than noise. Amplification is a *ratio* of whitened lengths, so widening the residual covariance along one direction can rotate two signatures apart as easily as together, and the pair that binds can change identity as it does. The claim checked above is the one the numbers support: the isolable sets do not move, and the hardest pair barely does.

## Units, and why they are not a detail here

The single-circuit loop in results/second_balance declared its states in kilograms and joules, which span seven orders of magnitude, and the kernel's positive-definiteness tolerance refused a prior declared honestly in them at two of three swept scales. Nondimensionalising is the fix that study named.

Every cell in this study is accepted at every swept prior. That is not a better result than the earlier study got; it is the same kernel being asked a question it can answer, and the difference is entirely a choice of units.

## The temperature datum, and why the row is written on the rise

Enthalpy is measured from an arbitrary zero, so no reported geometry may depend on where that zero is put. An earlier draft of this study carried the datum offset as the energy row's mass coefficient. The residual `A f` is genuinely datum-free — the offset cancels between the enthalpy a biased meter adds and the enthalpy the row subtracts — and that cancellation is exactly why the defect was easy to miss. What does not cancel is the row's own norm: `A P A^T` grows with the offset, so the whitening moves, and with it every angle reported from it.

That formulation is still built here, so the reason it was rejected is a number:

| datum below the supply | hardest pair, + header, 4 circuits |
| ---: | ---: |
| 0 | 1.26x |
| 1 | 1.40x |
| 2 | 1.52x |
| 5 | 1.67x |
| 20 | 1.74x |

A bookkeeping choice moves the answer by 1.38x over that range. Writing the row on the rise removes the offset from `A` entirely, so there is no datum left to move — and at offset 0 the rejected formulation reproduces the committed one exactly, which is what makes this a check rather than a comparison of two different models. The supply thermocouple's uncertainty survives as the variance of that now-zero coefficient, which is precisely what it is.

## What this does not establish

- A design study. Nothing here is measured on an installed manifold, and no fault is asserted to occur.
- Isolability is a property of the declared catalogue and the declared rows. A fault absent from the catalogue cannot be found, and the confounds reported are confounds of THIS catalogue.
- Structural isolability is not a detection guarantee. Every count in this report should be read with the tightest-pair amplification beside it, which is why none appears without one.
- Single-fault, static directions. Two circuits degrading together, and any temporal signature, are outside what this compares.
- The duty coefficient is one declared operating point. The effectiveness is swept rather than asserted, and the supply temperature's uncertainty is declared as A_var rather than assumed away, but the FORM of the duty relation is not varied.
- Fouling is declared as a process change that conserves energy. That it is a different KIND of thing from an instrument bias is an assumption of the catalogue, not a finding.
- The prior is declared, not measured. The heat-load axis it sweeps is the one the report finds binding, so a shop's real number for it matters more than anything else here.
