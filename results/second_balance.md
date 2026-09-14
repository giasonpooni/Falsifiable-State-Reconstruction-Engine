# Which second balance would let this repository isolate a fault

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 1baf7ee49925, git 4194bbfded. Latency columns are wall-clock on this machine and are not a claim.

A design study of PROPOSED constraint topologies. No instrument, record or estimate appears here; nothing below is a measurement.

Every constraint in this repository has rank(A) = 1, which means a scalar residual, which means every visible fault direction is collinear with every other. No fault can be isolated from any other at any covariance, with any filter, given any amount of data. This study asks what a second balance would change, before one is built.

## The result

| topology | rows | residual rank | faults isolable | structurally invisible | perfectly confounded pairs |
|---|---:|---:|---:|---:|---:|
| Ridgway closure (every constraint in this repository today) | 1 | 1 | **0 of 4** | 1 | 3 |
| Two-reach Muskingum river — *continuity only* | 2 | 2 | **1 of 7** | 1 | 4 |
| Two-reach Muskingum river — *continuity + declared routing* | 4 | 4 | **5 of 7** | 0 | 1 |
| Cooling loop, mass and energy over the same pipes — *mass + energy only* | 2 | 2 | **1 of 7** | 0 | 6 |
| Cooling loop, mass and energy over the same pipes — *mass + energy + declared duty* | 3 | 3 | **2 of 7** | 0 | 1 |

**Conservation alone buys almost nothing.** A second *conservation* row takes the river from 0 isolable faults to 1 and the cooling loop to 1, out of 7 declared candidates each. Adding a balance is not the same as adding information about which instrument moved.

**The constitutive relation is what separates sensors.** Muskingum's routing law makes storage and flow enter with different coefficients, and the cooling loop's duty relation involves the inlet temperature and not the outlet. With those rows the river reaches **5 of 7** and the loop **2 of 7**.

**The river design dominates, on two counts.** It isolates more than twice as many candidates, and what it still cannot separate is physical rather than instrumental:

- `inflow gauge bias` and `ungauged lateral inflow to reach 1` are the same direction. Water arriving that no gauge sees, and an inflow gauge reading high, are the same statement about the evidence. No topology fixes that; only a gauge on the lateral inflow does.

The cooling loop's remaining confounds are instrumental, which is worse:

- `outlet temperature bias` and `stored-energy sensor bias` are the same direction.

With conservation alone the loop leaves 6 pairs perfectly confounded: one energy equation gives one residual direction, so every fault that touches only the energy side collapses into it, whatever the sensors are.

## What routing makes visible

A common drift of all three flow gauges is in the null space of continuity — the classic invisible fault, the river's version of an equal-and-opposite bias. The routing rows weight the gauges differently, so it becomes visible and in fact isolable: it appears in `continuity only`'s invisible list and not in `continuity + declared routing`'s.

## The price: every constitutive row is a declared parameter

Routing rows carry K and x. The duty row carries effectiveness and the minimum heat-capacity rate. The kernel declares uncertainty on `b` through `b_var` and carries **none on `A`**, so the isolation a constitutive row buys is conditional on parameters whose uncertainty cannot currently be represented. Swept rather than assumed:

| Muskingum K (days) | x | residual rank | isolable |
|---:|---:|---:|---:|
| 0.5 | 0 | 4 | 5 of 7 |
| 0.5 | 0.2 | 4 | 5 of 7 |
| 0.5 | 0.35 | 4 | 5 of 7 |
| 0.5 | 0.5 | 4 | 5 of 7 |
| 1 | 0 | 4 | 5 of 7 |
| 1 | 0.2 | 4 | 5 of 7 |
| 1 | 0.35 | 4 | 5 of 7 |
| 1 | 0.5 | 4 | 5 of 7 |
| 2 | 0 | 4 | 5 of 7 |
| 2 | 0.2 | 4 | 5 of 7 |
| 2 | 0.35 | 4 | 5 of 7 |
| 2 | 0.5 | 4 | 4 of 7 |

| heat-exchanger effectiveness | residual rank | isolable |
|---:|---:|---:|
| 0.3 | 3 | 2 of 7 |
| 0.5 | 3 | 2 of 7 |
| 0.7 | 3 | 2 of 7 |
| 0.9 | 3 | 2 of 7 |

The river result is 5 of 7 across the declared range and drops to 4 at the corner of it. A design study that reported only its chosen parameter pair would not show that.

## Structural and conditional, kept apart

Which faults are isolable is structural: it does not move when the declared prior does, and this study checks that across a prior sweep of [0.01, 1.0, 100.0]. What does move is the amplification — how much larger an amplitude must be to isolate a fault than to detect one:

| topology | variant | tightest separated pair | amplification across the prior sweep |
|---|---|---|---|
| Ridgway closure (every constraint in this repository today) | one closure | none separated | — |
| Two-reach Muskingum river | continuity only | storage 1 bias vs middle gauge bias | 1.15x to 1.41x |
| Two-reach Muskingum river | continuity + declared routing | storage 1 bias vs storage 2 bias | 1.41x to 20.31x |
| Cooling loop, mass and energy over the same pipes | mass + energy only | inlet flow-meter bias vs outlet flow-meter bias | 6.08x to 6.08x |
| Cooling loop, mass and energy over the same pipes | mass + energy + declared duty | inlet flow-meter bias vs inlet temperature bias | 2.21x to 2.21x |

An amplification near 1 means a fault that can be detected can be named; a large one means the separation is real and useless. **Every separation in this study is actionable** -- the largest is 20.3x. An earlier draft of this study reported the cooling loop's tightest pair at 204,258x. That was a units error, not a finding: it whitened joules against kilograms by giving both unit variance. Declared in kilograms and joules the same pair reads 2.21x. A dimensionless prior is not a neutral choice.

**The cooling loop's own units are near the kernel's numerical limit.** Its states are kilograms and joules, which span enough orders of magnitude that a prior declared honestly in them is rejected by `check_spd`'s positive-definiteness tolerance at 2 of the 3 swept scales, in both variants. Those cells are reported as refusals rather than tuned around: the refused scales are listed per variant in the JSON. The river's states are all volumes and every declared prior is accepted. Nondimensionalising the loop would fix this, and is work the river design does not need.

## What this study does not show

- No instrument, record or estimate appears in this study; nothing here is a measurement.
- Isolability is structural. A topology that separates two faults here says the evidence could in principle name one, not that a real record would, at what magnitude, or how often.
- Every fault carries one unrestricted signed amplitude, so a direction and its negation are the same line; a sign convention cannot separate a pair this study calls confounded.
- Each constitutive row's coefficients are declared parameters. The kernel carries uncertainty on b through b_var and none on A, so the isolation a constitutive row buys is conditional on parameters whose uncertainty cannot yet be represented.
- Simultaneous faults, temporal signatures and unknown onset are outside what fdi.isolability scores.
- The cooling loop's states span seven orders of magnitude in natural units; a prior declared directly in those units can fail the kernel's positive-definiteness tolerance, which is a practical argument for nondimensionalising that design before building it.
