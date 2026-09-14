# Additive invariant filtering for fluid states

FSRT now has an invariant state/error layer for affine fluid models. It propagates
uncertainty, estimates the invariant error from measurements, and retracts that
correction onto the predicted state. It supports correlated measurements, missing
readings and fixed changes of state coordinates. It also retains the innovation
before correction for separate diagnostics.

This is the additive-group special case of invariant filtering. For these models it
is mathematically equivalent to an ordinary Kalman filter. That equivalence is an
acceptance requirement, not an accuracy improvement. General nonlinear Lie-group
IEKF models remain future work.

## Run it

```bash
uv run --frozen --python 3.13 python examples/invariant_fluid.py
uv run --frozen --python 3.13 python -m set_lcm.experiments.invariant_layer --quiet
```

The [example](../examples/invariant_fluid.py) estimates two tank masses and verifies
the same result with masses expressed in grams and a shifted coordinate origin. The generated
[experiment report](../results/invariant_layer.md) compares an independent ordinary
Kalman implementation with this layer on fixed synthetic records, including gauge
bias, correlated noise and dropout. Neither method receives simulated truth.

## What “invariant” means here

The state belongs to the additive group `(R^n, +)`. Use the error convention
`e = x - reference`, and retraction `x = reference + e`. Adding the same translation
to both states leaves their error unchanged. Left and right errors coincide because
addition is commutative.

For the declared affine model,

```text
x_next = F x + d + w       Cov(w) = Q
z      = H x + a + v       Cov(v) = R

predicted mean = F mean + d
predicted error = F e + w
innovation = z - (H predicted_mean + a) = H predicted_error + v
correction = K innovation
posterior mean = retract(predicted_mean, correction)
```

Error propagation is independent of the nominal trajectory. On an additive group,
the group-affine condition reduces to affine dynamics; this gives FSRT a concrete
instance of the geometry underlying IEKF methods. The general theory and its
stability conditions are given by [Barrau and Bonnabel](https://arxiv.org/abs/1410.1465).
Their nonlinear convergence results are not a blanket guarantee for arbitrary
fluid models or faulty measurements.

Conservation is a separate physical property. An exchange model may preserve total
mass through its transition and process-noise directions. Choosing an invariant
error does not itself impose a mass balance, establish a correct model, or make
confusable sensor faults identifiable. Existing reconciliation and diagnostic
functions remain separate operations.

## Public interface

```python
import numpy as np
from set_lcm.invariant import GaussianState, predict, update

state = GaussianState([40, 60], [[1, .2], [.2, 1.5]])
prior = predict(state, np.eye(2), [-.2, .2], [[.01, -.01], [-.01, .01]])
result = update(prior, [40.1, 60.4], np.eye(2), [[.25, .08], [.08, .36]])
state = result.posterior
```

`GaussianState` stores a finite mean and a full positive-semidefinite covariance.
`predict` requires explicit `F`, drift and `Q`. `update` accepts measurement values,
`H`, full `R`, an optional observation offset, and an optional boolean mask. Inputs
are copied into immutable state/results; inputs are not modified.

| Result field | Meaning |
|---|---|
| `prior`, `posterior` | Predicted and conditioned means/covariances. |
| `innovation`, `innovation_covariance` | Measurement disagreement and its full covariance before correction. |
| `gain`, `correction` | Mapping from observed innovation to estimated state error, and that correction. |
| `observed_indices` | Original channel indices in the order used by innovation and gain. |
| `statistic`, `dof` | Squared whitened innovation and observed dimension. |
| `status` | `updated` or `no_observations`; neither is a health verdict. |

Masked readings may be NaN; retained readings must be finite. The same mask selects
both measurement rows and the full principal submatrix of `R`. An empty/all-missing
record retains the prior with no statistic and zero observed dimension. `H`, offsets
and the entire declared covariance must still be valid. Complex values, infinities,
nonboolean masks and inconsistent dimensions are refused.

The covariance update uses Joseph form and standardized Cholesky solves. Singular
priors and process/measurement noise are supported when the resulting observed
innovation covariance is positive definite and numerically resolvable. Dependent
exact observations are refused; the layer does not invent extra degrees of freedom,
add jitter or silently use a pseudoinverse. Covariance validation uses correlation
coordinates so a large-unit row cannot hide an invalid small-unit variance.

## Coordinate consistency

`AffineCoordinates(T, c)` represents a fixed, known chart `x' = T x + c`. It transforms
the full state and model together:

```text
mean' = T mean + c           P' = T P T.T
F' = T F T^-1               d' = T d + c - F' c
Q' = T Q T.T               H' = H T^-1
a' = a - H' c
```

Use `transform_state`, `transform_dynamics` and `transform_observation` before
filtering in that chart; use `restore_state` to express the result in original units.
These helpers use solves and immutable arrays. They do not parse unit strings or
infer physical transformations. A translated origin changes the chart; it is not
an automorphism of the ordinary zero-identity additive group. Errors transform as
`e' = T e` without an additive offset.

The chart must be invertible, with a 2-norm condition number no greater than `1e12`.
This is a declared numerical support limit, not an identifiability bound or a promise
of a given accuracy near that limit. State/drift and covariance conversions additionally
check a round trip with tolerance `1e-8`: mean/drift errors use the larger of the input's
absolute value and standard deviation; covariance errors use input standard-deviation
products. Exactly zero scales allow no introduced error. This refuses destructive
origin cancellation and covariance underflow/overflow. A dense change of coordinates
can also make an exact singular covariance numerically unresolved; that case is refused
when its computed covariance fails validation, without jitter or projection.

These guards do not guarantee precision in every later model calculation. The experiment
checks a specified set of charts, not every invertible matrix. Time-varying charts and uncertain coordinate calibrations require
different propagation and are outside this helper's contract.

Changing measurement coordinates separately also requires changing the measurement,
`H`, offset and full `R`. Reordering/scaling individual channels preserves missingness
when the mask is reordered. General mixing must first select the observed subspace;
reusing the original mask after mixing channels is invalid. The innovation statistic
is invariant to invertible changes of observed coordinates. A Gaussian log density
has a Jacobian term and does not retain the same numerical value after unit changes.

## Evidence and limits

Tests compare a complete record against independent joint Gaussian conditioning,
cross-check the existing linear KF, retain deterministic covariance directions, and
check prediction/update consistency under coordinate changes. Synthetic Gaussian
draws check innovation and marginal interval calibration under the declared model.
The generated experiment also checks that coordinate equivalence persists when an
injected gauge bias violates that model; this does not establish robustness to bias.

The filter assumes known affine matrices/offsets, declared covariances, process noise
independent of the current state error, and measurement noise independent of the
predicted error. Cross-channel covariance is supported; cross-time or shared-reference
errors need explicit state augmentation or a joint-record treatment such as the
[fluid measurement baseline](FLUID_BASELINE.md). Gaussian calibration of the innovation
statistic additionally requires the Gaussian model to hold. There is no automatic
uncertainty fitting, fault exclusion, operational alarm policy or arrival-time scheduler
inside this layer. Call it in the correct sample sequence with the appropriate interval
model. It does not turn interval-mean storage into boundary measurements.

The next IEKF step requires a specific nonlinear fluid model with a derived group
structure, autonomous invariant-error dynamics where applicable, a compatible
observation model, and comparison with an ordinary EKF on the same evidence. A new
retraction alone would not establish those properties. The immediate experimental
priority remains independently referenced fluid measurements and characterized faults.
