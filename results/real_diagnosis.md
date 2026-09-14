# What the diagnostic surface says about a real record

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 e65c60119ce2, git 1da70ef98a (source dirty). Latency columns are wall-clock on this machine and are not a claim.

The diagnostic surface applied to real records, with the candidate catalogue declared in each site's own TOML. No fault is asserted to exist at either site.

`diagnose()` is the most developed decision machinery here: candidate fits with amplitude intervals, a nuisance space, a minimum separation between candidates, and explicit `ambiguous` and `insufficient_evidence` outcomes. Every record it had judged until now was one this repository generated. `real_fluid_baseline` calls it on a real record but with an **empty** hypothesis set — a consistency test with no candidates at all. The isolation half had never met real evidence.

`results/real_taylor_park` produced what was missing: a real record whose balance does not close, by a margin that is not arguable. This asks the obvious next question of it.

## The catalogue is declared, not written here

Each site's candidates live in its own TOML. A fault's *signature* cannot: it is what the candidate would do to this particular residual, so it depends on how many days each window held and what the gauges saw. The declaration names a profile and this study resolves it — the same split as a constraint row whose right-hand side is a reading.

**Taylor Park Reservoir, Taylor River, Colorado** — 5 candidates:

| candidate | amplitude unit | what it is |
| --- | --- | --- |
| `ungauged_constant` | acre-ft per day | Constant unmeasured inflow |
| `outflow_reads_low` | ft^3/s | Outlet gauge under-reporting by a constant |
| `seasonal_creeks_unreported` | ft^3/s | Seasonal creeks flowing while not reporting |
| `storage_scale_error` | fraction | Stage-capacity table off by a scale factor |
| `inflow_rating_high` | fraction | Inflow ratings high by a common fraction |

**Ridgway Reservoir, Uncompahgre River, Colorado** — 4 candidates:

| candidate | amplitude unit | what it is |
| --- | --- | --- |
| `ungauged_constant` | acre-ft per day | Constant unmeasured inflow |
| `outflow_reads_low` | ft^3/s | Outlet gauge under-reporting by a constant |
| `storage_scale_error` | fraction | Stage-capacity table off by a scale factor |
| `inflow_rating_high` | fraction | Inflow ratings high by a common fraction |

The first site declares no seasonal candidate because it has no gaps for one to live in: every Ridgway series is complete. That difference is not a convenience — it is what makes anything separable at the second site at all, as the next table shows.

## The structural result, which no covariance can change

Two candidates a practitioner thinks of as entirely different things — water no gauge sees, and an outlet gauge reading low — do the same thing to a closure residual: they add a constant volume per day. They are declared separately because they *are* different physically, and on this residual they are **exactly the same direction**.

| site | pair | cos |
| --- | --- | ---: |
| taylor_park | `ungauged_constant` vs `outflow_reads_low` | -1.0000 |
| ridgway | `ungauged_constant` vs `outflow_reads_low` | -1.0000 |

This is the rank-1 bound arriving in a real catalogue. One closure row gives a scalar per window, so every pair of candidates it can see at all is collinear in that one direction; no covariance, no amount of data and no threshold choice separates them. A catalogue holding only those two could never resolve, at any site, ever. Reporting that is the engine working — the alternative is naming one of them and being right half the time.

What breaks the tie at Taylor Park Reservoir is the record's own gaps. `seasonal_creeks_unreported` follows the count of absent days rather than the calendar, so it points somewhere else:

| pair | cos |
| --- | ---: |
| `seasonal_creeks_unreported` vs `ungauged_constant` | +0.6478 |
| `seasonal_creeks_unreported` vs `outflow_reads_low` | -0.6478 |
| `seasonal_creeks_unreported` vs `storage_scale_error` | -0.0770 |
| `seasonal_creeks_unreported` vs `inflow_rating_high` | -0.0998 |

## What the engine says, against how much unmodelled error is admitted

The covariance is the declared instrument sigmas, propagated through the window sum. Nothing is fitted to the residual's own scatter — that would be circular. But those sigmas describe *instruments*, and say nothing about the daily-mean alignment error this repository has documented as present and unquantified from the start. Picking a value for it would be tuning, so it is declared per day and swept.

**Taylor Park Reservoir, Taylor River, Colorado** — cumulative residual 63,472 acre-ft over 36 windows:

| alignment sd (acre-ft/day) | null χ² | threshold | status | candidates not rejected |
| ---: | ---: | ---: | --- | ---: |
| 0 | 11,523.4 | 58.6 | `unexplained` | 0 |
| 50 | 1,309.3 | 58.6 | `unexplained` | 0 |
| 200 | 108.9 | 58.6 | `ambiguous` | 4 |
| 500 | 18.1 | 58.6 | `consistent` | 5 |
| 1,000 | 4.5 | 58.6 | `consistent` | 5 |

**Ridgway Reservoir, Uncompahgre River, Colorado** — cumulative residual 1,526 acre-ft over 36 windows:

| alignment sd (acre-ft/day) | null χ² | threshold | status | candidates not rejected |
| ---: | ---: | ---: | --- | ---: |
| 0 | 442.4 | 58.6 | `unexplained` | 0 |
| 50 | 136.8 | 58.6 | `unexplained` | 0 |
| 200 | 20.3 | 58.6 | `consistent` | 4 |
| 500 | 3.6 | 58.6 | `consistent` | 4 |
| 1,000 | 0.9 | 58.6 | `consistent` | 4 |

Read the second site's column downward and the engine changes its mind in exactly the way it should. With instrument sigmas alone the record is **unexplained** — not by one candidate, by all of them; the declared uncertainty is far too tight for anything in the catalogue to account for what the gauges did. Admit more unmodelled error and several candidates become adequate *at once*, which is **ambiguous**: something is there and the engine cannot tell you which. Admit enough and the record stops needing an explanation at all — `consistent`, with nothing to diagnose.

**At no point does it name a single cause.** That is the honest outcome for a rank-1 constraint and a catalogue containing two collinear members, and it was predicted before the table was computed. An engine that named one here would be wrong in a way that is hard to detect and expensive to trust.

The first site sits below the second at every sweep point — its balance closes, so there is less to explain — which is the same ordering its closure residual and its ungauged estimate give, by the same evidence.

## What this does not establish

- No fault is asserted to exist at either site. A candidate being adequate means the record does not reject it, which is not evidence that it happened.
- The catalogue is what each site declares. A cause absent from it cannot be found, and the ambiguity reported here is a property of the catalogue and the constraint, not a property of the reservoir.
- One closure row is rank 1, so its residual cannot separate candidates that act on it identically. That is a structural bound, not a limitation of the data.
- The alignment term is swept, not measured. Nothing here establishes its true size; the report says what the diagnosis is at each declared value and no more.
- Windows are equal-length divisions of the record, not calendar months, so seasonality is approximate at the window edges.
- Amplitude intervals are conditional on the single-candidate model and on the declared covariance at that sweep point.
