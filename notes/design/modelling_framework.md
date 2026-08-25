# A Fitting Framework for JACT Intensity Models

## 1. Purpose and scope

This document describes a scalable JAX fitting layer for intensity models that
will be consumed by JACT.

JACT already owns the downstream multi-state runtime:

- `StateSpace` defines states and allowed directed transitions,
- `Model` binds intensity callables to that topology,
- `solve()` computes duration-dependent state probabilities and expected cashflows,
- `simulate()` samples event histories from the same discretized intensities.

The fitting layer should not duplicate those responsibilities. Its job is to
turn interval-split event-history data into fitted, pure-JAX intensity
callables that satisfy JACT's public contract.

The desired boundary is therefore:

```text
event-history data
        -> fitted parameters and preprocessing state
        -> JACT-compatible intensity callable(s)
        -> StateSpace.build(...)
        -> solve(...) / simulate(...)
```

This distinction matters for package design. JACT currently has only JAX and
JAXlib as runtime dependencies and accepts models fitted by any framework. A
trainer may use Optax, Equinox, Flax, Arrow, Polars, or other optional tools,
but those should not become requirements of the exported inference artifact or
of JACT's core runtime.

The initial fitting framework should prioritize:

- a statistically correct competing-risks likelihood,
- explicit agreement with a `jact.StateSpace`,
- stable and smooth hazards on JACT's time-duration grid,
- fixed-shape, JIT-friendly training,
- sharing information across related transitions,
- a compact inference cost inside `solve()` and `simulate()`,
- and reproducible reconstruction of a JACT `Model`.

Throughout this document, clock time is written \(t\), current source-state
duration is written \(d\), and an allowed transition from state \(r\) to state
\(s\) has intensity

\[
\lambda_{rs}(t,d,z)=\exp\{f_\theta(r,s,t,d,z)\},
\]

where \(z\) denotes other covariates. The exponential is a modelling choice;
the JACT boundary receives non-negative intensities, not log-intensities.

---

## 2. JACT contracts that constrain the design

### 2.1 Topology is owned by `StateSpace`

The fitting configuration should be constructed from, or validated against,
the same `jact.StateSpace` used for valuation and simulation. JACT:

- rejects self-transitions,
- permits absorbing states and cyclic graphs,
- requires every declared transition to be assigned exactly once when a model is built,
- orders states by their declaration order,
- orders exits from one source by target-state order,
- and reduces a solve or simulation to states reachable from the declared initial-state set.

The fitter should derive one deterministic edge table from
`state_space.ordered_transitions()` and store the state-space fingerprint with
the artifact. It should not invent an independent transition ordering.

Training may omit loss contributions for transitions that are structurally
impossible from a row's source state. Export still has to cover every
transition declared by the `StateSpace`, including transitions unreachable
from a particular deployment initial state.

### 2.2 The inference callable is a function of `(t, d, **kwargs)`

JACT's single-transition protocol is conceptually:

```python
intensity(t, d, **kwargs) -> value broadcastable to (batch, D)
```

At solve time:

- `t` is scalar-like clock time for the current solver step,
- `d` is a duration grid with shape `(1, D)`, with duration on the last axis,
- each non-scalar covariate in `kwargs` has leading shape `(batch, ...)`,
- scalar covariates are constants and do not determine batch size,
- and a callable output does not determine batch size.

Useful single-transition output shapes are `()`, `(D,)`, `(1, D)`,
`(batch, 1)`, and `(batch, D)`. A per-individual hazard should normally use
`(batch, 1)`, not `(batch,)`, because the latter is generally not broadcastable
over the duration axis.

The source state is normally encoded by which transition assignment JACT is
evaluating. It need not be passed as a dynamic solver covariate. The fitting
model may use integer source and destination identifiers internally, but its
adapter must bind those identifiers into JACT callables.

### 2.3 Grouped assignments have a leading transition axis

JACT can bind intensities in three ways:

| Assignment | Coverage | Output convention |
|---|---|---|
| `transitions={(r, s): fn}` | one edge | broadcastable to `(batch, D)` |
| `exits={r: fn}` | every exit from \(r\) | leading axis in `state_space.targets(r)` order |
| `groups={fn: edges}` | the listed edges | leading axis in the supplied edge order |

`jact.wrappers.bind_intensity`, `bind_exit_intensity`, and
`bind_grouped_intensity` adapt a separate feature function and model apply
function to these protocols. The grouped wrappers move the selected model
output axis to the front and clamp outputs with `maximum(output, 0)`.

A natural fitted model can emit `(batch, D, K)` hazards and use
`output_axis=-1`; the resulting JACT callable emits `(K, batch, D)`.

### 2.4 JACT uses midpoint-discretized semi-Markov dynamics

For `horizon=H` and `steps_per_unit=m`, JACT uses step width

\[
h=1/m
\]

and a duration grid of width

\[
D=H m.
\]

The probability solver transports duration density, evaluates intensities at
cell midpoints, aggregates all exits from a source as competing risks, and
resets duration to zero on entry to a destination state. It also tracks exact
initial point masses with per-individual starting duration and evaluates them
along \(d_0+t\).

The simulator uses the same rectangular, midpoint-evaluated intensity
approximation. Its sampled event times are continuous inside those cells, but
the hazard surface is still discretized by `steps_per_unit`.

Consequently, fitting an accurate continuous surface is necessary but not
sufficient. The exported model must also behave well under JACT's midpoint
evaluation and at the intended deployment resolution.

---

## 3. Canonical event-history data

The natural training observation is one interval during which a subject is at
risk in one source state. A row should contain at least

\[
(i, r, t_0, t_1, d_0, d_1, z, e),
\]

where:

- \(i\) identifies the independent trajectory or subject,
- \(r\) is the state occupied over the interval,
- \([t_0,t_1]\) is the clock-time interval,
- \([d_0,d_1]\) is the source-state duration interval,
- \(z\) contains covariates valid over the interval,
- \(e\) is the observed transition edge at \(t_1\), or no event.

For an uninterrupted at-risk interval,

\[
t_1-t_0=d_1-d_0.
\]

Duration resets at an observed transition. Keeping clock time and source-state
duration as separate columns is essential: JACT intensity callables receive
both `t` and `d`, and attained age, calendar effects, and time since entry are
not interchangeable.

Use an edge identifier rather than only a destination identifier in the
canonical batch. This makes the ordering explicit and avoids accidental
confusion when batching rows from multiple source states. The edge identifier
must index the deterministic edge table derived from the `StateSpace`.

Additional useful fields include:

- sampling or case weights,
- cluster identifiers,
- left-truncation metadata,
- missingness indicators,
- covariate validity timestamps,
- and a mask for padded rows in fixed-size batches.

The row likelihood is conditional on survival to \(t_0\), so ordinary delayed
entry can be represented by beginning exposure at the entry time. More complex
observation mechanisms, interval-censored transitions, simultaneous events,
or uncertain states require explicit likelihood extensions and should not be
silently coerced into this row format.

Units are part of the model contract. If JACT will be called with time in
years, training exposure, duration, spline knots, age transformations, and
reported hazards must all use years. Store the unit in artifact metadata.

---

## 4. Competing-risks likelihood

Let \(\mathcal O(r)\) be the exits from source state \(r\). The exact row
contribution is

\[
\ell_i =
\mathbf 1(e_i\neq\varnothing)
\log \lambda_{e_i}(t_1^-,d_1^-,z(t_1^-))
-
\int_0^{\Delta_i}
\sum_{s\in\mathcal O(r_i)}
\lambda_{r_i s}(t_0+u,d_0+u,z(t_0+u))\,du,
\]

where \(\Delta_i=t_1-t_0\).

The superscript \(-\) means immediately before the transition. Event-induced
changes to state or covariates must not leak into the event hazard.

This must be the core objective. Fitting independent event-versus-no-event
classifiers per edge does not reproduce the continuous-time likelihood and can
mishandle exposure and competing exits.

For a piecewise-constant row evaluated at representative covariates,

\[
\ell_i =
\mathbf 1(e_i\neq\varnothing) f_{e_i}
-
\Delta_i
\sum_{s\in\mathcal O(r_i)}\exp(f_{r_i s}).
\]

This covers both event and censored rows without constructing negative samples.
The loss implementation should evaluate all declared edge scores for a batch,
mask them by row source state, sum only valid exits, and gather the event edge
for the event term.

Event terms should remain in log-hazard space. Computing a hazard and then
taking its logarithm is avoidable and can underflow. The JACT adapter can apply
the non-negative link only at inference.

If rows carry observation weights \(w_i\), define clearly whether the optimized
quantity is a weighted population likelihood, a correction for sampling, or a
frequency likelihood. Do not mix those interpretations.

---

## 5. Likelihood integration and agreement with JACT

Two fitting backends are useful, but they should not be confused with JACT's
runtime solver.

### 5.1 Piecewise-exponential fitting

The initial backend should treat the intensity as constant over each training
row. It is efficient, easy to stream, and appropriate when rows are already
split at covariate changes and on a sufficiently fine time scale.

Use a documented representative point, preferably the row midpoint

\[
(t_0+\Delta/2, d_0+\Delta/2),
\]

so the approximation has the same interpretation as JACT's midpoint cells.
The training split grid need not equal the JACT solve grid, but validation must
measure sensitivity to both.

### 5.2 Fixed quadrature fitting

When hazards change materially within a row, approximate the cumulative hazard
with fixed Gauss-Legendre quadrature. Fixed nodes are vectorizable,
differentiable, and shape-stable under JIT.

This is a fitting feature only. JACT does not currently perform 4- or 8-point
Gaussian quadrature inside each solver cell; it uses one midpoint evaluation.
A more accurate training integral therefore does not eliminate JACT
discretization error.

### 5.3 Deployment convergence is a separate test

For each intended use case, run JACT at increasing `steps_per_unit` and compare:

- state probabilities,
- material terminal cashflows,
- event-incidence summaries,
- and, when relevant, duration-distribution summaries.

Smooth hazards should show second-order midpoint convergence. Discontinuities
inside cells may converge more slowly. Known jumps in clock time or duration
should be aligned with the JACT grid when practical.

---

## 6. Training-time model interface

The fitting core should use a dense, topology-aware interface such as

```python
log_hazards(params, batch) -> (batch_size, n_edges)
```

or, for quadrature,

```python
log_hazards(params, quadrature_batch) -> (batch_size, n_nodes, n_edges)
```

All edges use the deterministic `StateSpace` edge order. Static arrays map:

- edge id to source-state id,
- edge id to target-state id,
- source-state id to a padded list of outgoing edge ids,
- and padded slots to a Boolean validity mask.

This representation makes the competing-risks sum cheap and gives a single
place to implement parameter sharing. It is preferable to training one Python
callable per transition.

This is deliberately not the public JACT callable interface. Training is
row-based, while JACT inference is grid-based and binds state/edge identity
through `StateSpace.build()`. A small adapter should bridge the two.

---

## 7. Recommended parameterization

A strong default is a structured model with a shared nonlinear residual:

\[
f_{rs}(t,d,z)
=
b_{rs}
+g^{\mathrm{time}}_{rs}(t)
+g^{\mathrm{duration}}_{rs}(d)
+g^{\mathrm{add}}_{rs}(z)
+a_{rs}^{\top}h_\phi(t,d,z,r).
\]

### 7.1 Edge intercepts

Each allowed transition receives an intercept \(b_{rs}\). Initialize it from
the empirical event rate

\[
b_{rs}^{(0)}\approx
\log\frac{N_{rs}+\epsilon}{T_r+\epsilon_T},
\]

where \(N_{rs}\) is the observed event count and \(T_r\) is exposure in the
source state.

### 7.2 Clock-time and duration effects

Clock time and current-state duration should be separate terms. Depending on
the application, clock time may be replaced or augmented by attained age or
calendar date, but the exported feature builder must reconstruct it from `t`
and solve-time covariates.

Spline bases are a good default for effects that must remain smooth under
JACT's grid evaluation. Curvature penalties such as

\[
P(\beta)=\gamma\sum_j(\Delta^2\beta_j)^2
\]

help stabilize sparsely observed regions.

### 7.3 Explicit effects

Linear, categorical, and scientifically specified interaction terms should be
easy to add. Categorical vocabularies, reference levels, unknown-category
handling, and missingness rules belong in the fitted artifact.

### 7.4 Shared nonlinear representation

A shared representation

\[
h_\phi(t,d,z,r)\in\mathbb R^q
\]

can feed edge-specific or low-rank output heads. This lets rare transitions
borrow information from common transitions while retaining edge heterogeneity.
Origin-state embeddings are usually enough initially; destination or edge
embeddings can be added if the data support them.

The neural component should remain compact. In JACT, an intensity model is
evaluated repeatedly over a `(batch, duration)` grid inside a time scan, not
just once per portfolio member. A network that is cheap during row-based
training may dominate an actuarial solve.

### 7.5 Partial pooling

A useful decomposition is

\[
f_{rs}=f^{\mathrm{shared}}_r+f^{\mathrm{specific}}_{rs},
\]

with stronger regularization on the edge-specific component. Evaluate pooling
by source-state family or clinically/actuarially meaningful transition groups;
do not assume all edges should share equally.

---

## 8. From log-hazards to valid JACT intensities

The training model should produce finite log-hazards. The inference adapter
must produce finite, non-negative hazards in the same time unit as the solver.

Possible links include:

- `exp(log_hazard)`, with explicit control of the supported log range,
- `softplus(raw)` for a less explosive positive link,
- or a structured positive baseline multiplied by bounded relative effects.

Blind clipping can make extrapolation appear stable while hiding a bad model,
so bounds should be justified and tested. In particular:

- `exp` can overflow on extrapolated features,
- the JACT wrappers clamp negative values but do not repair NaNs or positive infinity,
- the probability solver currently prevents negative integrated hazards by clamping them to zero,
- and the simulator rejects non-finite or materially negative intensities
  unless a wrapper has already hidden the negative output.

Artifact validation should therefore call the unwrapped apply function and the
wrapped JACT callable on in-domain and stress-test grids. Reject non-finite
outputs; do not rely on downstream clamping as model validation.

---

## 9. JACT adapter and model construction

The fitted artifact should expose separate feature construction and application
functions. A typical per-source adapter is:

```python
import jax.numpy as jnp
import jact


def healthy_features(t, d, *, entry_age, smoker):
    batch = entry_age.shape[0]
    attained_age = jnp.broadcast_to(
        entry_age[:, None] + t,
        (batch, d.shape[-1]),
    )
    duration = jnp.broadcast_to(d, attained_age.shape)
    smoker_grid = jnp.broadcast_to(smoker[:, None], attained_age.shape)
    return jnp.stack([attained_age, duration, smoker_grid], axis=-1)


def apply_healthy_exits(params, x):
    # Model-specific implementation; last axis follows StateSpace target order.
    log_hazard = structured_model(params, x)
    return jnp.exp(log_hazard)


healthy_exits = jact.wrappers.bind_exit_intensity(
    apply_healthy_exits,
    fitted_params,
    healthy_features,
    output_count=2,
    output_axis=-1,
)

model = state_space.build(
    exits={"healthy": healthy_exits},
    transitions={
        ("disabled", "dead"): disabled_mortality,
    },
)
```

Before binding an exit model, assert that its output labels exactly match
`state_space.targets(source)`. For arbitrary pooling groups, store the ordered
edge list with the artifact and pass that same list to `groups={fn: edges}`.

One global all-edge predictor can be bound as a `groups` assignment covering
`state_space.ordered_transitions()`. Per-source `exits` assignments are often
easier because source identity is static and the output contains no irrelevant
edges.

### 9.1 Current grouped-evaluation limitation

The grouped API expresses the right statistical model, but current JACT solve
and simulation paths do not have identical evaluation behavior:

- the simulator preserves grouped assignment blocks and evaluates a grouped
  callable once before selecting its edge outputs,
- the probability/cashflow solver currently stores per-edge slice wrappers and
  invokes the grouped callable separately for each covered edge, and may also
  evaluate a point-mass path separately.

Therefore, `exits` or `groups` does not currently guarantee one shared neural
forward pass per solver step. This is important for the proposed shared
representation.

If grouped fitted models are a primary use case, JACT should preserve grouped
blocks in `ReducedModel` and let the solver evaluate each block once for the
density grid and once for any required point values, then distribute selected
hazards to edges. The simulation plan already demonstrates the relevant block
representation. Until then, benchmark the actual `model.solve()` workload and
keep grouped networks small.

---

## 10. Artifact boundary and reconstruction

A fitted artifact should contain:

- parameter PyTree arrays,
- preprocessing constants,
- spline knots and basis metadata,
- categorical vocabularies,
- ordered states and edges or a topology fingerprint,
- output labels and their order,
- feature names, dtypes, and units,
- supported input ranges,
- static architecture configuration,
- and fitting/version metadata.

Its inference path should require only JAX operations and should be pure,
deterministic, JIT-compatible, and differentiable where the chosen model is
differentiable.

JACT serializes a `StateSpace` topology, but a built `Model` contains Python
callables and is not itself a portable serialized fitted model. The fitting
framework must provide an explicit loader that:

1. loads and validates artifact metadata,
2. reconstructs preprocessing and apply functions,
3. validates the supplied `StateSpace` and edge order,
4. creates wrapper callables,
5. and calls `state_space.build(...)`.

Do not pickle an opaque closure and treat that as the long-term interchange
format. Store arrays and declarative metadata, with a versioned reconstruction
path.

Training convenience objects, optimizer state, data loaders, and validation
history may be checkpointed separately. They should not be required for
valuation or simulation.

---

## 11. Feature processing and extrapolation

Preprocessing is part of the model, not a notebook prelude. The artifact should own:

- numeric centering and scaling,
- transformations such as log age or capped duration,
- spline knots and boundary behavior,
- categorical encodings and unknown levels,
- missing-value rules,
- time-origin conventions,
- and time units.

The feature builder must operate on JACT shapes without Python loops over batch
or duration. It will commonly turn `t`, `(1, D)` duration, and `(batch, ...)`
covariates into `(batch, D, features)`.

JACT forwards the same `kwargs` container at every solver step. Time-varying
features must therefore be derived from `t` and `d`, or read from a fixed-shape
exogenous covariate path with JAX indexing. A callable cannot mutate portfolio
covariates after a transition. Transition-dependent changes should instead be
represented by the destination state's callable and the reset duration. Models
that require endogenous history beyond current state and duration need an
expanded state space or a different runtime abstraction.

Training support and deployment support differ. JACT may query:

- later clock times than appear in training,
- durations near the full solve horizon, or as large as
  `initial_duration + horizon` along an initial point-mass path,
- non-zero initial durations,
- states with little observed exposure,
- and covariate combinations created by a valuation portfolio rather than the training sample.

Boundary behavior must be intentional. Reasonable choices include linear spline
tails, bounded effects, or documented rejection of unsupported portfolios.
Silent high-order polynomial extrapolation is unsuitable for repeated hazard
integration.

---

## 12. Optimization defaults

A practical trainer can start with:

- AdamW,
- gradient clipping,
- learning-rate warmup and decay,
- early stopping on subject-level validation likelihood,
- explicit smoothness penalties,
- and exponential-moving-average or checkpoint-averaged parameters.

Empirical edge-rate initialization and near-zero nonlinear residuals should put
the initial model close to a constant-intensity competing-risks fit.

Negative log-likelihood remains the primary objective, but model selection
should also consider:

- edge-specific calibration,
- smoothness along `t` and `d`,
- maximum and minimum predicted hazards on deployment grids,
- JACT solve convergence,
- and downstream error in material probabilities or cashflows.

A small likelihood improvement is not worthwhile if it creates unstable tails
or makes every JACT solve several times more expensive.

---

## 13. Sampling, splitting, and very large data

Split at the independent subject or trajectory level. Intervals from one
trajectory must not appear in both training and validation. Depending on the
deployment question, also consider temporal, geographic, institutional, or
policy-cohort holdouts.

Unbiased shuffled row minibatches are the simplest training default. If rare
events motivate stratified sampling, weight rows by inverse sampling
probability so the stochastic objective still estimates the desired full-data
likelihood.

For tens of millions of intervals, stream fixed-shape batches:

```text
partitioned columnar data
        -> CPU batch construction
        -> prefetch
        -> device transfer
        -> jitted loss and update
```

Pad the last training batch and use a row-validity mask. Shape changes can
trigger extra compilation. Data loading should remain independent of topology,
model application, likelihood integration, and optimization.

Data-parallel multi-device training is likely sufficient because the proposed
models should be small enough to replicate. Multi-host and model-parallel
training are later concerns, not version-one requirements.

This is distinct from JACT deployment batching. Repeated `solve()` calls should
also keep batch size, horizon, `steps_per_unit`, `record_every`, output
structure, and device selection stable. Large portfolios should be padded to a
fixed solve batch and trimmed after evaluation.

---

## 14. JACT runtime performance implications

The deployment cost has a different shape from the fitting cost. A JACT solve
has a time scan of length \(H m\), and transition callables are evaluated over
a duration grid whose width is also \(H m\). A grid-dependent neural model can
therefore incur work proportional to the product of time and duration
resolution, in addition to batch and edge dimensions.

Design and benchmark for this actual workload:

- broadcast constant effects rather than materializing them early,
- calculate shared covariate transformations once where the current callable boundary permits,
- keep spline and MLP widths modest,
- avoid transition-specific full networks,
- use per-source outputs when a global edge output wastes substantial work,
- and measure compile time, peak memory, and steady-state execution.

`record_every` reduces stored output, not the number of inner hazard updates.
Reducing probability output to `StateProbability()` or disabling it for
cashflow-only work saves output memory but does not make an expensive intensity
surface cheap.

JACT can shard the individual batch over local devices. That does not shard the
duration grid or eliminate repeated grouped evaluation in the current solver.

---

## 15. Numerical behavior and dtype

Hazards must be finite, non-negative, and expressed per solver time unit. Test
both hazard values and integrated step hazards

\[
h\lambda(t,d,z).
\]

Large integrated hazards can be mathematically valid, and JACT uses stable
exponential-transfer formulas, but a coarse cell with a rapidly changing or
very large hazard can still be a poor midpoint approximation.

At least float32 should be used for likelihood accumulation and log-hazard
calculation. Float64 may be appropriate for long-horizon or precision-sensitive
actuarial work, but JAX x64 must be enabled and the complete solve inputs must
carry the intended dtype. JACT's solver value dtype is inferred from floating
initial-distribution and covariate inputs; captured model parameters alone
should not be assumed to select the solver dtype.

Numerical validation should vary dtype and `steps_per_unit` separately. A
resolution error is not repaired by float64, and a roundoff problem is not
necessarily repaired by a finer grid.

Use stable primitives in the loss and link functions, including masked
log-hazard gathers and log-domain calculations where appropriate. Monitor NaNs,
infinities, gradient norms, and edge-specific extrema during training.

---

## 16. Validation at three boundaries

### 16.1 Statistical validation

Evaluate out-of-sample:

- total and per-edge negative log-likelihood,
- cumulative incidence and state occupancy calibration,
- rare-edge behavior,
- temporal and duration calibration,
- and relevant subgroup performance.

### 16.2 Artifact validation

For every exported group:

- verify state and edge labels and order,
- compare training-model and exported-model hazards on random inputs,
- test scalar, `(batch, 1)`, and full duration-grid broadcasting,
- test JIT and autodiff when required,
- stress supported input boundaries,
- and reject non-finite or negative outputs.

### 16.3 JACT integration validation

Build an actual JACT model and test:

- complete, non-overlapping transition coverage,
- all relevant declared initial states and non-zero initial durations,
- probability conservation and non-negativity,
- convergence as `steps_per_unit` increases,
- agreement between simulated incidence and solved probabilities within Monte Carlo error,
- stability of material cashflows,
- fixed-size deployment batches,
- and local multi-device execution if it will be used.

Because JACT reduces to the reachable graph, test more than the most common
initial state. A transition can be valid and fitted yet never exercised by one
particular integration test.

---

## 17. Proposed software boundaries

A fitting companion can be organized as:

```text
TopologySpec
    derived from jact.StateSpace
        |
EventHistoryDataset -> CanonicalBatch
        |                  |
        |             FeatureEncoder
        |                  |
        +----------> LogHazardModel
                           |
                    LikelihoodBackend
                    /              \
          piecewise midpoint    fixed quadrature
                           |
                     Trainer / Optax
                           |
                     FittedArtifact
                           |
                       JACTAdapter
                           |
              StateSpace.build(...)
                    /              \
                 solve          simulate
```

The stable separations are:

\[
\boxed{
\text{topology}
\neq\text{data}
\neq\text{features}
\neq\text{log-hazard model}
\neq\text{likelihood}
\neq\text{optimizer}
\neq\text{JACT adapter}
}
\]

The adapter is a first-class component, not an afterthought. It is where edge
ordering, feature broadcasting, positive linking, artifact validation, and
wrapper selection meet JACT's runtime contract.

Given JACT's deliberately small dependency surface, the fitting implementation
should initially live as a separate package, an optional extra, or a clearly
isolated experimental module. It should not make Optax or a neural-network
library mandatory for users who only solve with hand-written intensities.

---

## 18. Recommended version one

Version one should include:

1. A topology specification derived from `StateSpace` ordering.
2. Canonical interval rows with clock-time and duration bounds.
3. Integer source-state and event-edge identifiers.
4. Subject identifiers, exposure, row weights, and padding masks.
5. A piecewise-exponential competing-risks likelihood evaluated at row midpoints.
6. Empirically initialized edge intercepts.
7. Standardized continuous and encoded categorical predictors.
8. Separate smooth clock-time and duration effects.
9. A small shared SiLU MLP with low-rank edge heads.
10. Explicit smoothness and weight regularization.
11. Optax training with fixed-shape streaming batches.
12. A versioned artifact containing preprocessing and topology metadata.
13. Per-source JACT exit adapters plus arbitrary group adapters.
14. Automated hazard parity and shape tests.
15. End-to-end `solve()` and `simulate()` convergence tests.
16. Benchmarks on realistic JACT batch, horizon, and duration-grid sizes.

Before making the neural component larger, improve JACT's solver-side grouped
evaluation if benchmarks show repeated shared forward passes dominate runtime.

---

## 19. Later extensions

Once the basic boundary is stable, useful extensions include:

- fixed-quadrature training likelihoods,
- monotone or shape-constrained smooths,
- richer partial-pooling structures,
- destination and edge embeddings,
- transition-specific low-rank adapters,
- interval-censored event likelihoods,
- left-truncation helpers and observation-process models,
- uncertainty estimates and ensembles,
- temporal cross-validation and automated calibration reports,
- multi-host data-parallel training,
- mixed-precision neural components with higher-precision likelihoods,
- and a block-aware JACT solver representation for single-pass grouped inference.

Direct generator-matrix output is not a priority. JACT's public abstraction is
an intensity callable over clock time, source-state duration, and portfolio
covariates; JACT owns competing-risk aggregation, duration transport,
reachability reduction, simulation, and cashflow integration.

---

## 20. Central design principle

The statistical core should predict all declared edge log-hazards jointly:

\[
(r,t,d,z)\longmapsto
\{f_{rs}(t,d,z):s\in\mathcal O(r)\}.
\]

The deployment core should expose those predictions through JACT's actual
callable and topology contracts:

\[
(t,d,\text{solve-time covariates})
\longmapsto
\text{non-negative hazards broadcastable to }(B,D).
\]

Keeping both sides explicit aligns:

- the correct competing-risks likelihood,
- information sharing across transitions,
- source-state duration semantics,
- stable midpoint evaluation,
- JAX vectorization,
- artifact portability,
- and direct use in JACT probabilities, simulations, and cashflows.

The intended result is not a replacement for JACT and not a trainer embedded
inside every JACT solve. It is a fitting companion whose final product is a
small, validated, JACT-native intensity model.
