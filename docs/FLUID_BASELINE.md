# Fluid-measurement baseline

This baseline checks whether storage and flow measurements agree over matching time
intervals, propagates their joint uncertainty, and compares declared fault explanations.
It is validated against analytical identities and controlled synthetic experiments.
It has **not** been validated as an industrial sensor diagnosis system.

Run the small, offline example:

```bash
uv run --frozen --python 3.13 python examples/fluid_baseline.py
```

It returns `consistent` for conserving measurements, `identified` for a sufficiently large
storage step, and `ambiguous` for storage drift that a flow offset can also explain. The
fault profiles and their onset are supplied. This example does not search an operational
stream for unknown events.

## Measurements and interval support

[`BalanceRecord`](../src/set_lcm/measurement.py) contains contiguous interval edges in
seconds, storage values, flow channels, signs (`+1` in, `-1` out), covariance and optional
evidence IDs. It makes immutable copies of the numerical inputs.

| Input | Required interpretation |
|---|---|
| `storage_support="instant"` | One storage reading at every interval boundary: `n+1` readings for `n` intervals. |
| `storage_support="mean"` | One mean storage per interval; explicitly declare `within_interval_model="constant_net_flow"`. |
| `flow_support="mean"` | One mean rate per interval and channel, in `m3/s`. |
| `flow_support="total"` | One integrated volume per interval and channel, in `m3`. |
| `volume_unit` | Explicitly `"m3"`; convert other units in an adapter. |
| `covariance` | Variance vector or full joint covariance, in raw-measurement order. Unknown uncertainty is not zero. |
| `evidence_ids` | Optional ID or group of IDs per raw measurement. Reused evidence does not become independent. |

Raw order is all storage readings followed by `flows.ravel(order="C")`: interval first,
channel second. A full covariance can represent sensor correlations, temporal correlation,
or a shared calibration reference. Each entry has the product of its two component units.
The API checks numerical validity and supported unit declarations; it cannot verify a
physical uncertainty model. A variance vector explicitly declares independent raw errors.

For boundary storage and mean net flow `q`, the exact interval residual is

```text
r[k] = S[k+1] - S[k] - dt[k] q[k]
```

For mean storage, under constant net flow within each interval, the correct residual is

```text
r[k] = Sbar[k+1] - Sbar[k]
       - (dt[k] q[k] + dt[k+1] q[k+1]) / 2
```

These formulas also support unequal interval durations. Total flows already contain the
duration factor. Mean storage alone cannot establish the within-interval model; the API
refuses mean storage without its explicit assumption. General within-interval dynamics
need a different averaging operator or finer measurements.

This is an offline calculation. Output times label interval ends for boundary storage and
the midpoint of the second interval for mean storage. They are **not availability times**.
The record must be complete; adapters must split into complete contiguous windows or reject
incomplete records. Dropping readings must not silently bridge an interval gap.

## One operator for the residual and its uncertainty

For raw measurements `y`, `balance_residuals(record)` builds the balance operator `H`:

```text
r = H y
C = H Cov(y) H.T
```

`balance_residuals(record, cumulative=True)` applies cumulative summation to the same
operator. The first storage error is therefore shared by all cumulative residuals.
Intermediate storage errors cancel where the algebra cancels their measurements; IDs of
canceled measurements disappear from that residual's contributing evidence.

With independent boundary storage errors of variance `sigma_S²` and independent interval
volume errors of variances `v[k]`, cumulative residuals numbered from one satisfy

```text
Cov(R[i], R[j]) = sigma_S² + sum(v[k], k < min(i,j))  for i != j
Var(R[i])      = 2 sigma_S² + sum(v[k], k < i)
```

The common first reading contributes the first term, once in the joint covariance. It must
not be treated as fresh independent evidence at each step. Adjacent and cumulative forms
give the same joint Mahalanobis statistic when their covariance is nonsingular: invertible
summation cannot create information. The tests verify this identity and covariance against
independent Gaussian draws.

## Conditional fault explanations

[`diagnose`](../src/set_lcm/diagnostics.py) accepts a residual vector, positive-definite
joint covariance and a dictionary of residual signatures. A signature is the response to
one unit of a specified fault profile, after applying the measurement operator. The fitted
amplitude may have either sign. Its physical unit follows from the supplied signature.

An optional nuisance matrix describes unbounded deterministic effects such as a constant
ungauged inflow. The method whitens with the covariance, projects out the nuisance span,
then fits each remaining candidate by generalized least squares. A candidate absorbed by
the nuisance span is marked unobservable. Allowing an arbitrary nuisance at every time
removes all diagnostic information and returns `insufficient_evidence`.
Numerically unresolved nuisance rank is refused rather than silently removing a possible
explanation. Rescale or reparameterize such a model before fitting it.

The reference test uses chi-square degrees of freedom `n-rank(N)` for no fault and
`n-rank(N)-1` for an observable single-amplitude candidate. Quantiles are numerically
inverted; tests compare them with independent SciPy reference values. SciPy is not a
runtime dependency. The assumptions are known Gaussian covariance, fixed signatures and
onset, and at most one candidate fault plus the declared nuisance.

| Outcome | Meaning |
|---|---|
| `consistent` | The no-fault model is not rejected. Invisible or small faults may still be present. |
| `identified` | No-fault rejected; exactly one supplied candidate is adequate. Conditional on this catalogue and model. |
| `ambiguous` | No-fault rejected; multiple supplied candidates remain adequate. |
| `unexplained` | No-fault and all testable supplied candidates rejected. |
| `insufficient_evidence` | Too little residual information remains to make the required comparison. |

An empty catalogue performs only a consistency check and assigns no fault cause. Singular
residual covariance is refused by the diagnostic API; exact/PSD measurement covariances are
supported by the measurement operator. Supporting them in diagnosis would first require
checking the deterministic constraints: a nonzero residual in a zero-variance direction
is a contradiction, not a component to discard. Only then can the uncertain part be reduced
to independent coordinates. This reduction is not implemented by `diagnose()`.

The fitted amplitude interval is Gaussian and **conditional on that candidate being the
model**. It is not a probability that the candidate caused the fault, and it is not adjusted
for choosing a candidate or searching event times. The test level is per fixed record,
not an operational alert rate or a joint probability over candidate tests. A model-external
cause can resemble a unique supplied candidate; `identified` is not causal proof.

## Reproducible evidence

```bash
uv run --frozen --python 3.13 python -m set_lcm.experiments.fluid_baseline --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_fluid_baseline --quiet
uv run --frozen --python 3.13 --dev pytest -q tests/test_measurement.py tests/test_diagnostics.py tests/test_fluid_baseline.py tests/test_real_fluid_baseline.py
```

The [synthetic report](../results/fluid_baseline.md) covers offset, drift and gain at two
predeclared magnitudes, varying and constant flow, conserving physical changes, omitted
flow, nuisance-confounded faults and a common storage offset. Each case has 16 development
and 128 independent evaluation seeds. No parameter is fitted on either set; the separation
is a reproducible benchmark convention, not external validation. Seeds are reused across
cases for paired comparison, so cases must not be pooled as independent trials. The report
retains per-seed fits, misses, ambiguity, attribution errors and interval coverage with
explicit denominators. It evaluates one fixed horizon per record, not detection delay.

The [Ridgway report](../results/real_fluid_baseline.md) uses the committed real measurements,
their evidence IDs and the existing consumer-declared uncertainty sweep. Daily storage
means are matched to daily flows under an explicit constant-net-flow assumption. Fixed,
nonoverlapping windows retain shared-reference covariance. Missing windows and omitted
cross-window pairs are counted. Real sensor truth is unavailable: rejection is a
conditional inconsistency, and the report assigns no sensor cause. Its cross-day raw
measurement covariance is declared zero, not empirically calibrated.

Both reports have JSON companions, generator provenance and default-suite reproduction
checks. Same-build numerical reproduction is exact; cross-build values use a declared
relative allowance of `1e-8`, while statuses, counts and structure must match exactly.
The allowance is an engineering tolerance, not a measurement of universal platform error.

The historical `wb_open`, `wb_aug` and `wb_closed` filter experiment retains its documented
interval/reference approximation. The new baseline is a separate measurement calculation;
it does not silently recalibrate those filters or their published results.

## What the next experiment must add

Unknown onset, simultaneous faults, changing covariance, timing faults, nonlinear rating
curves and uncertain dynamics remain unvalidated. Before operational use, measure alert
episodes, misses, delay and wrong attribution on reviewed field events or a controlled
fluid loop with independent reference instruments. Thermal modeling follows when an
experiment supplies the measurements needed to validate the energy balance.
