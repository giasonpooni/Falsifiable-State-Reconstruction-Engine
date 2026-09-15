# A second independent balance: the two-reach Muskingum design

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v33-x86_64-with-glibc2.39; source sha256 33e4e335962b, git ec69c5a08c. Latency columns are wall-clock on this machine and are not a claim.

Simulated two-reach Muskingum records with known injected faults. Synthetic throughout: no field validation and no real record.

The design study predicted from the algebra that continuity alone would isolate almost nothing and that Muskingum's own storage relation would take this topology to 5 of 7 declared candidates. This builds the system and tests that against simulated records. The rule is the residual vector and its joint covariance through `diagnose()`; adjacent intervals share a storage reading and `Cov(r) = M R Mᵀ` carries that exactly.

## What each fault does to one interval's rows

| declared direction | continuity 1 | continuity 2 | routing 1 | routing 2 |
|---|---:|---:|---:|---:|
| storage 1 sensor bias | +0.00 | +0.00 | +1.00 | +0.00 |
| storage 2 sensor bias | +0.00 | +0.00 | +0.00 | +1.00 |
| inflow gauge bias | -1.00 | +0.00 | -0.20 | +0.00 |
| middle gauge bias | +1.00 | -1.00 | -0.80 | -0.20 |
| outflow gauge bias | +0.00 | +1.00 | +0.00 | -0.80 |
| common flow-gauge drift | +0.00 | +0.00 | -1.00 | -1.00 |
| ungauged lateral inflow | -1.00 | +0.00 | -0.20 | +0.00 |
| coordinated storage/flow offset (K:1) | +0.00 | +0.00 | +0.00 | +0.00 |

Three things are visible in that table alone. **The storage sensors move only the routing rows** — continuity differences the two storage readings, so a persistent bias cancels exactly, which is why continuity alone cannot see them at any magnitude. **An ungauged lateral inflow produces the same residual as an inflow-gauge bias**, so no covariance and no length of record separates them. And **a coordinated offset of both storage sensors by K times a common offset on all three flow gauges produces exactly zero**: the topology's blind spot, named.

## Recovery, on the evaluation seeds

64 evaluation seeds, disjoint from the 16 development seeds, at both declared magnitudes. Cells are the fraction of records where the rule identified the injected fault and named it correctly; a wrong name is counted separately in the JSON and is not in these cells.

| injected fault | continuity only, 1.5σ | routing, 1.5σ | continuity only, 4σ | routing, 4σ |
|---|---:|---:|---:|---:|
| storage 1 sensor bias | 0% | 5% | 0% | 100% |
| storage 2 sensor bias | 0% | 9% | 0% | 100% |
| inflow gauge bias | 12% | 17% | 98% | 100% |
| middle gauge bias | 19% | 17% | 100% | 100% |
| outflow gauge bias | 11% | 14% | 100% | 100% |
| common flow-gauge drift | 0% | 48% | 0% | 100% |
| *healthy river (false alarms, α = 0.01)* | 2% | 0% | 2% | 0% |

Routing is what makes the design worth building, and the storage rows are where the difference lives: continuity cannot see either storage sensor or a common flow drift at 4 sigma any more than at 1.5, because it cannot see them at all. The weaker column is the one that says this design is not magic — at 1.5 sigma it names the right instrument in a minority of records.

## A physical cause the catalogue does not contain

An ungauged lateral inflow is injected: the river changes and every instrument is honest. With only instrument candidates declared, the rule reports **identified — and names an instrument — in 100% of seeds**. It is not wrong about the evidence; the two directions are identical. It is wrong because the catalogue offered it no other answer.

Declaring the physical explanation as a candidate converts that into **100% ambiguous with the correct explanation among them**. The lesson is about the catalogue, not the topology: a catalogue of only instrument faults will confidently name an instrument for a river.

## Routing rows carry measured parameters

The isolation above is conditional on K and x. Running the same healthy river with the true reach at K = 1.15, x = 0.28 while the rows still declare K = 1, x = 0.2:

| declaration | false-alarm rate on a healthy river |
|---|---:|
| parameter error, A treated as exact | **100%** |
| parameter error, uncertainty declared | 0% |

A mis-declared reach reads as a fault in every record until the uncertainty in its parameters is declared. That is stage 3b's `A_var` doing the job it was built for, on the first topology that needed it.

## What this does not show

- Synthetic records from the same Muskingum model the rows declare, except where parameter error is injected on purpose. Agreement is not field validation.
- One hydrograph, one reach pair, one onset, two magnitudes. A different wave or onset could order these differently.
- The onset is declared and supplied; searching for it is a different and harder problem, and the amplitude intervals diagnose() reports are conditional on the supplied profile.
- An ungauged lateral inflow and an inflow-gauge bias produce the same residual direction. No covariance and no amount of record separates them; only another gauge does.
- A coordinated offset of both storage sensors by K times a common offset on all three flow gauges is in the null space and produces exactly zero residual, at any magnitude.
- Parameter uncertainty is declared as a linear sensitivity in K and x about the declared values, first order in the parameter error like the A_var term it mirrors.
