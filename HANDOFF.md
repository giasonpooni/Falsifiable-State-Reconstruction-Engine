# Handoff — 2026-09-12

The build moves from the local session that wrote this repository to the cloud
session "FSRE assembly and build". This file is for a session starting cold. README.md
is the map of what exists and what each result does and does not show; this file says
what was in progress, what is decided, and how the work has been done.

## State at handoff

`main` = everything verified to date: the Phase 1 slice and its hardening, P2 (per-sensor
CUSUM channel; augmented filter `kf_aug` with pump scale α and boundary flux L;
baselines `kf_closedq`, oracle bound, fed-back projection), P2b's first item (`b_var`,
declared uncertainty on `ConstraintSet`), the runner decoupled from `Truth`
(`PublicInputs`; an AST test pins it), the DAF bridge (`src/set_lcm/bridge/daf.py`), and
the first truth-free real-data report (`results/real_noaa.md`, NOAA 8454000, one day).

- Fast tests: `uv run --python 3.13 --dev pytest -q` (~10 min, 146 tests).
- Slow tests: `uv run --python 3.13 --dev pytest -q -m slow` (~25 min). The reproduction
  test regenerates the grid and asserts it equals `results/summary.json` value for value.
- DAF-gated tests (2) run when `DAF_ROOT` points at a DAF checkout at commit `6b37859`
  (vendored substrate `5e146d5`): DAF's own code recomputes all 728 evidence ids in
  `data/daf/`, and the export tool reproduces those files byte for byte.

## Stopped before it committed anything

A five-stage run ("fluid-water-balance") was stopped on the user's instruction. None of
its stages had committed or written to either repository. Its plan is the next work, in
this order.

1. **NOAA 8454000, one month.** A live fetch of 2024-01-01 → 2024-01-31 (MLLW, metric,
   GMT) through DAF's own NOAA adapter. The user approved this fetch. Commit the raw bytes
   so offline re-export is byte-identical. Compare the 240 readings of 2024-01-15 value by
   value with the committed one-day fixture. Then run a tidal filter the month can
   support: M2, S2, N2, K1, O1, M4; state what the Rayleigh criterion still cannot
   separate, e.g. K1/P1. Fit q on days 1–15, and score each of days 16–31 separately.
2. **A USGS daily-values adapter and extractor in DAF.** Follow DAF's own conventions:
   see the NOAA and USGS-earthquake adapters and the PHASE_17/31/32/34 docs. Approval
   status and qualifiers are revision metadata and stay out of content, as NOAA's `q`
   does. There is no per-value uncertainty, so none is emitted and none is defaulted.
   Add a PHASE_45 doc.
   **The user does not want anything pushed to the DAF repository.** Write the work in a
   DAF checkout on a branch, and carry it into this repository as a `git format-patch`
   series against `6b37859`, together with the fixtures DAF's extractor exports.
3. **Generalise the bridge beyond NOAA.**
   - Declare series by a selector.
   - Make time semantics explicit: `instant` (the current NOAA path, unchanged) or
     `calendar_day`, with a declared zone and a declared anchor.
   - When the source states no uncertainty, require consumer-declared absolute or
     relative σ, with a citation.
4. **The reservoir water balance through the reconciliation kernel.**
   - Make conservation a static linear row the kernel checks, via cumulative volumes:
     S_k − S_0 − c·Σ(V_in − V_out) = 0, with c = 86400/43560 acre-ft per ft³/s-day.
   - Mirror the testbed with three filters: `wb_open` + projection/guard/`b_var`/feedback;
     `wb_closed` (closure in the dynamics); `wb_aug` (ungauged net inflow L as a state).
   - Add a model-free daily closure residual computed from the data alone.
   - Report d(f), including the equal-and-opposite inflow/outflow gauge bias that the
     balance cannot see.
   - Keep the report truth-free, with an explicit "cannot validate" list.
5. **Final review**, then regenerate every `results/` file under one source hash.

## Site selection (done, unverified research)

Full output: `docs/handoff/usgs_site_selection_2026-09-12.json`. It came from 33 requests to
`api.waterdata.usgs.gov`, all within the user's approval. The numbers below are one
agent's descriptive, in-sample checks. No committed code or test reproduces them yet.

**Chosen: Ridgway Reservoir, Uncompahgre River, Colorado.**

| role | USGS series | parameter / statistic |
|---|---|---|
| storage | USGS-09147022 | 00054 acre-ft, statistic 00003 = daily **mean** |
| outflow (just below the dam, no separate spillway gauge) | USGS-09147025 | 00060 ft³/s, daily mean |
| inflow | USGS-09146200 Uncompahgre R nr Ridgway, 149 sq mi | 00060, daily mean |
| inflow | USGS-09147000 Dallas Ck nr Ridgway, 97.2 sq mi | 00060, daily mean |

- Gauged fraction 246.2 / 265 sq mi = **0.929**. The ungauged 7.1% is Beaver Creek plus the
  lake-local area.
- Common window: WY2023–WY2025 (2022-10-01 → 2025-09-30, 1,096 days). No gaps, all values
  Approved, 0 revisions.
- Descriptive closure: with centred alignment, the residual has mean 0.73 cfs and SD 20.6
  cfs, against about 170 cfs of throughput. The 3-year cumulative imbalance is −0.4% of
  inflow. Monthly mean residuals run from −46 cfs (snowmelt, June 2025) to +47 cfs
  (monsoon, August 2023).

Caveats that shape the design:

- **Storage is a daily mean.** It pairs with daily-mean flows only half a day out of step,
  so use centred alignment: the residual SD is 20.6 cfs centred, against 27.4 cfs
  same-day.
- **Winter ice.** Inflow values flagged ESTIMATED: Dallas Ck 290 of 1,096, Uncompahgre 82.
- **No evaporation data.** It is unmeasured, and the lake-area figure behind the "few cfs"
  estimate is from memory; it needs a citation or has to be fitted.
- **Undefined daily date.** The API defines the daily `time` only as "the date an
  observation represents", with no zone. Legacy NWIS practice is the site's local
  standard-time day, but this API does not state it. DAF must either declare it as a
  cited assumption or refuse to assign UTC day bounds.
- **API mechanics:**
  - Base URL: `api.waterdata.usgs.gov/ogcapi/v1`.
  - Paging is by cursor; `limit` goes up to 10,000.
  - Items arrive in no time order.
  - `value` is a string.

**Runners-up:** Horseshoe + Bartlett, Verde River, AZ (gauged fraction 0.951, end-of-day
storage, desert evaporation). San Carlos, AZ (0.970). Kanopolis, KS (0.965).
Canyon Lake, TX was rejected: its storage series ended in 2002.

## Roadmap agreed with the user ("fit this for fluid dynamics")

1. **One reservoir** (above).
2. **A river with several gauges**, with Muskingum routing between them (still linear).
3. **A pipe or cooling network.** First, the rank-deficient incidence matrix: dependent
   rows are currently refused when `b_var` is declared, and need the Schur-complement
   reduction. Then nonlinear head loss by iterated projection, which gives
   `NOT_CONVERGED` its first producer. Then inequalities (QP / OSQP).
4. **Scale beyond a dense P** for large networks; continuum flow fields need
   ensemble or reduced-order estimators.

## How the work has been done (standing rules the user agreed)

- `results/` is a verified artifact, and every results file carries a source-tree hash.
  A change to `src/` means regenerating the results and running the slow suite.
- When a change should not move numbers, diff `results/*.json` against the parent
  (excluding `latency_us_*` and provenance) and state "N pre-existing values changed".
- Every README number traces to `results/`, and every README result also says what it
  does not show.
- Identified parameters are fitted on one window and evaluated on a disjoint one. Never
  tune `KFConfig`, `AugConfig`, `ClosedQConfig` or seeds to look better.
- Real-data reports are truth-free.
- Never overwrite an observation or the unprojected estimate.
- The estimator side never reads hidden truth. The only exception is the labelled oracle
  bound, which gets it through an explicit argument.
- Network access only for sources the user approved.
- Nothing is pushed to DAF.
- Each stage is: implement → adversarial verification → fix commits. No amend, no force.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## DAF facts a new session needs

- Repository: https://github.com/atomtrapping/Data-Acquisition-Channel. It has **no
  `main`**: the default branch is `claude/daf-state-space-architecture-v8peqd`.
- Boundary contract: `docs/DAF_STATE_SPACE_BOUNDARY.md` §§10–13, invariants I1–I8.
- CI gate (`.github/workflows/conformance.yml`):
  - `python -m pytest -q`: about 1,600 tests. Five need full git history, so never run
    the gate on a shallow clone. Tests marked `network` make live calls to EDGAR, arXiv,
    USGS and NOAA; locally, run with `-m "not network"`.
  - Doctrine regeneration: `epistemics.doctrine.write()`, then a clean
    `git diff docs/generated/`.
  - The vendored substrate must be unmodified.
  - `mypy` over `daf/ science/ boundary/ bridge/ epistemics/ session/ commerce/ tools/`.

## Known nits

- Per-seed latency values churn the results JSON on every regeneration.
- The fast suite takes about 10 minutes and could be split into tiers.
- No licence has been chosen.
