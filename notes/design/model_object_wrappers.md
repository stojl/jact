# Design note: exporting fitted intensities into Jact

## Status

The generic wrapper layer described in the first version of this note is
implemented in `jact.wrappers`. This revision records the actual public API and,
more importantly, defines the export boundary that a fitted intensity model
must satisfy to work correctly in Jact.

The normative API remains `docs/api_spec.md`. This note is design guidance for
the fitting-to-Jact handoff and identifies follow-up work; it is not a second API
specification.

---

## Context

Jact consumes transition intensities through pure solver callables:

```python
fn(t, d, **kwargs) -> array_like
```

The fitting design in `notes/design/modelling_framework.md` uses a different,
training-oriented abstraction:

```python
log_intensities(params, batch) -> [batch_size, max_outgoing]
```

That distinction is useful. Training should be free to use interval rows,
padded risk sets, integer state identifiers, and log-intensities. Jact, however,
evaluates a model repeatedly on a clock-time-by-state-duration grid and binds
each returned intensity to a declared transition. A fitted artifact therefore
needs a small export adapter; the training predictor should not be handed to
`StateSpace.build(...)` unchanged.

The export adapter has four responsibilities:

1. convert Jact coordinates and solve-time covariates into fitted-model
   features;
2. convert log-intensities or unconstrained scores to actual intensities;
3. remove padding and put outputs in Jact's transition order; and
4. expose a pure, deterministic, JIT-compatible callable with Jact's shape
   semantics.

`jact.wrappers` handles the framework call and output-axis normalization. The
fitted artifact remains responsible for feature semantics, the link from model
score to intensity, and transition identity.

---

## Decision: the final exported quantity is a Jact intensity

The fitting layer may model

\[
f_{rs}(x) = \log \lambda_{rs}(x),
\]

but its Jact-facing `apply_fn` must return \(\lambda_{rs}(x)\), not
\(f_{rs}(x)\). Jact interprets every returned value as an instantaneous rate in
the time unit used by `horizon`, `steps_per_unit`, clock time, and duration.

A suitable exported apply function is conceptually:

```python
def apply_rates(params, features):
    log_rates = predict_log_intensities(params, features)
    return jnp.exp(log_rates)
```

An intrinsically positive parameterization such as `softplus` is also valid.
The choice of link belongs to the fitted model specification and must be the
same at training and inference time.

The wrappers currently return `jnp.maximum(output, 0.0)`. This is a final
safety clamp, not a substitute for a positive output link:

- it changes the meaning of a log-intensity if one is passed accidentally;
- it gives zero gradient for negative raw outputs;
- it can hide a poorly calibrated or broken model from the simulator's
  negative-intensity validation; and
- it does not repair `NaN` or infinite outputs.

The exported intensity should therefore already be non-negative and finite on
the complete domain Jact will evaluate. Any stabilization of `exp`, bounds on
extrapolation, or maximum-rate policy must be chosen and tested by the fitting
artifact rather than silently invented by Jact.

---

## Actual public wrapper API

The wrappers live in the public `jact.wrappers` submodule. They are deliberately
not top-level aliases.

```python
jact.wrappers.bind_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    model_state=None,
    apply_kwargs=None,
)

jact.wrappers.bind_grouped_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    output_count,
    output_axis=-1,
    model_state=None,
    apply_kwargs=None,
)

jact.wrappers.bind_exit_intensity(
    apply_fn,
    params,
    feature_fn,
    *,
    output_count,
    output_axis=-1,
    model_state=None,
    apply_kwargs=None,
)
```

All three return ordinary callables accepted by `StateSpace.build(...)`.
`bind_exit_intensity()` is an intent-revealing alias over the grouped wrapper;
it does not receive a `StateSpace` or a source state and cannot validate the
topology itself.

The call made by a single wrapper is:

```python
features = feature_fn(t, d, **kwargs)
raw = apply_fn(params, features, **apply_kwargs)
```

When `model_state` is not `None`, it is instead:

```python
raw = apply_fn(
    {"params": params, **model_state},
    features,
    **apply_kwargs,
)
```

This supports frozen inference collections such as Flax batch statistics. In
the current implementation:

- `model_state` must not contain its own `"params"` entry because it would
  replace the explicit `params` value;
- a Flax apply function that expects `{"params": params}` even without other
  collections needs either `model_state={}` or a small explicit adapter;
- mutable inference that returns `(output, updated_state)` is not supported;
  the wrapper output must be one array-like intensity value; and
- `feature_fn` may return a PyTree if `apply_fn` accepts it, but the model output
  itself may not be a PyTree.

`apply_kwargs` is copied into the closure and is intended for static inference
configuration such as `train=False`, not per-solve data. Per-solve values belong
in `**kwargs` and are processed by `feature_fn`.

---

## Jact coordinate and shape contract

### Inputs seen by `feature_fn`

Jact supplies:

- `t`: scalar clock time;
- `d`: current-state duration values; and
- `**kwargs`: scalar or batch-major solve-time covariates.

For the continuous density calculation, `d` normally has shape `(1, D)`. For
point-mass calculations, including non-zero initial duration, Jact may evaluate
the same callable with per-individual durations shaped `(batch, 1)`. A feature
builder must support both and should use broadcasting rather than assume that
the first axis of `d` is always one.

For example:

```python
def feature_fn(t, d, *, baseline_age, sex_code):
    age = jnp.asarray(baseline_age)
    sex = jnp.asarray(sex_code)
    if age.ndim == 0:
        age = age[None]
    if sex.ndim == 0:
        sex = sex[None]

    attained_age, duration, sex = jnp.broadcast_arrays(
        age[:, None] + t,
        d,
        sex[:, None],
    )
    return {
        "numeric": jnp.stack([attained_age, duration], axis=-1),
        "sex": sex,
    }
```

Only axis 0 of a non-scalar solve kwarg is a batch axis. Callable outputs never
define Jact's batch size, and all non-scalar kwargs must agree on the size of
axis 0. Solve-time covariates must be numeric and convertible with
`jnp.asarray`; encode strings and categories before calling Jact. The names
`initial` and `initial_duration` are reserved and cannot also be feature
covariates. If a model needs static preprocessing constants, spline knots,
vocabularies, state identifiers, or feature scaling, those should be stored in
the fitted artifact or closure rather than repeated as batch covariates.

Clock time and duration have distinct meanings:

- `t` is elapsed time since the start of the solve;
- `d` is time since entry into the current state and resets after a transition;
- attained age is normally `baseline_age + t`, not `baseline_age + d`; and
- calendar time is normally `baseline_calendar_time + t`.

Training and export must use the same time unit. A model trained with days of
exposure cannot be used with a Jact horizon measured in years without an
explicit conversion of both coordinates and rates.

### Single-transition output

`bind_intensity()` accepts any model output broadcastable with Jact's eventual
`(batch, D)` solver grid. Useful shapes include:

- scalar `()`;
- duration-only `(D,)` or `(1, D)`;
- individual-only `(batch, 1)`; and
- full `(batch, D)`.

The wrapper does not force the result to `(batch, D)`; the solver performs the
authoritative broadcast once its batch size is known.

### Grouped and exit output

A grouped model must include an output axis. The wrapper moves `output_axis` to
the front, checks that its size equals `output_count`, and leaves every selected
output broadcastable to `(batch, D)`.

Consequently, all of these can be useful layouts:

- `(batch, D, K)` with `output_axis=-1`;
- `(K, batch, D)` with `output_axis=0`; and
- `(K,)` with `output_axis=0` for constant transition-specific rates.

The contract is not limited to rank-three arrays. Shape checks that depend on
the actual feature output run when the wrapper is called or traced, not through
an eager dummy call at wrapper construction. A dummy call would be unreliable
because Jact does not know the required covariates, batch shape, dtype, or model
input PyTree at construction time.

---

## Mapping a fitted transition graph to Jact

The fitting artifact must persist a stable mapping between its integer state or
edge identifiers and the state names used to construct `StateSpace`. Matching
only on array position is too fragile for a serialized model.

Every Jact transition must be covered exactly once across `transitions`,
`exits`, and `groups`. The three assignment modes have different ordering
rules.

### Separate transition models

Use `transitions` when the exported apply function returns one rate surface:

```python
onset = jact.wrappers.bind_intensity(
    onset_apply_rates,
    onset_params,
    feature_fn,
)

model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset,
        # Every other declared transition must also be assigned exactly once.
    }
)
```

### One packed model per source state

This is the closest match to the fitting framework's
`[batch, max_outgoing]` predictor. Build one fixed-source adapter for each
transient state and return only its valid, unpadded outputs in this exact order:

```python
targets = state_space.targets("healthy")
# Equivalent transition order:
# state_space.exits("healthy") == tuple(("healthy", x) for x in targets)

healthy_exits = jact.wrappers.bind_exit_intensity(
    healthy_apply_rates,
    fitted_params,
    feature_fn,
    output_count=len(targets),
    output_axis=-1,
)

model = state_space.build(exits={"healthy": healthy_exits, ...})
```

`StateSpace.targets(source)` orders targets by their order in
`StateSpace.states`, which may differ from the packed training graph. The
source-specific adapter must gather/reorder outputs and discard padded slots
before returning them. `bind_exit_intensity()` cannot do this automatically.

### One model for an arbitrary transition group

For `groups`, output position `i` corresponds exactly to transition `i` in the
supplied list:

```python
ordered_edges = [
    ("healthy", "disabled"),
    ("healthy", "dead"),
    ("disabled", "dead"),
]

all_rates = jact.wrappers.bind_grouped_intensity(
    all_edges_apply_rates,
    fitted_params,
    feature_fn,
    output_count=len(ordered_edges),
)

model = state_space.build(groups={all_rates: ordered_edges})
```

An origin-conditioned predictor cannot directly serve a cross-origin group
unless its export adapter evaluates all listed origins and assembles one global
edge vector. A source-specific `exits` adapter is usually simpler.

---

## Other Jact-specific considerations

### Purity and inference state

Intensity and feature functions must be deterministic, side-effect-free, and
JIT-compatible. Disable dropout and use frozen normalization statistics.
Python data-dependent branching, host NumPy conversion, pandas/sklearn
preprocessors, callbacks, and mutation do not belong inside the exported
function. Preprocessing needed for inference should be re-expressed in JAX and
stored with the fitted artifact.

### Repeated evaluation and smoothness

`solve()` evaluates intensities repeatedly at midpoint clock times and over a
duration grid. `simulate()` uses the same model on midpoint cells and samples
continuous event times under the resulting rectangular piecewise-constant
approximation. The fit should therefore be stable and reasonably smooth over
the full reachable time-duration domain, including extrapolation beyond the
training support.

Very large rates may remain mathematically valid but make results highly
sensitive to `steps_per_unit`. Export tests should include a refinement check,
not only pointwise prediction accuracy.

### Grouped-model performance

Grouped assignment currently has different evaluation behavior in Jact's two
engines:

- `simulate()` evaluates a grouped callable once and slices all requested
  outputs;
- `solve()` builds a single-transition closure for each edge, so the traced
  JAXPR contains one grouped-call expansion per selected transition.

In a representative grouped MLP test, the current XLA optimizer eliminated
those identical expansions: optimized HLO contained the same number of shared
matrix multiplications and activations for one, two, and three exits. Thus this
is not a demonstrated runtime duplication problem for ordinary pure models.

It is still a compiler optimization rather than a structural guarantee made by
Jact. Effectful custom calls, non-identical adapters, or future compiler changes
could prevent common-subexpression elimination. Benchmark an expensive or
unusual shared model end to end. A future solver change could represent grouped
evaluation as one explicit block, matching the simulator and making sharing
independent of compiler optimization.

### Differentiation and parameter lifetime

The wrappers capture `params` and `model_state` in a Python closure. That is a
good inference boundary for a finished fit. Replacing captured parameters means
constructing a new wrapper/model and may lead to another compilation.

For repeated sensitivities with respect to model parameters, parameters are
better supplied as dynamic JAX inputs through a deliberately designed callable
or the model/wrapper must be constructed inside the transformed function. The
fitting loop should not use `Model.solve()` as its ordinary minibatch training
loop.

### Multi-device execution

With `devices >= 2`, Jact shards the individual batch. `feature_fn` and
`apply_fn` see only a local `(local_batch, ...)` shard of each non-scalar
covariate; scalar covariates remain scalar. Jact may pad by repeating the last
real individual, and removes padding from public results. Exported inference
must be row-local and must not depend on global batch statistics, global batch
indices, or a particular batch size.

### Numerical validation differs by engine

The exported model must return finite non-negative rates for both engines.
`simulate()` explicitly rejects non-finite and materially negative intensities,
whereas solver arithmetic may propagate non-finite values into its result. The
current wrappers clamp all negative raw values before either engine sees them,
so tests must inspect the exported rates directly rather than rely only on an
end-to-end simulation error.

Use at least float32 for the final link and intensity arithmetic. If a neural
trunk uses lower precision, convert before `exp`/`softplus` and before returning
rates when needed for numerical stability.

### Serialization

Jact serializes `StateSpace`, but not `Model`, wrapper closures, fitted
parameters, preprocessing, or framework state. A deployable fitted artifact
must serialize those pieces separately and reconstruct:

1. the exact state and transition mapping;
2. static preprocessing and architecture information;
3. parameters and immutable inference state;
4. fixed-source or global-edge apply adapters; and
5. the Jact wrappers and `Model`.

Loading should reject an artifact whose saved topology or time-unit metadata
does not match the target `StateSpace`.

---

## Validation and testing plan

The export boundary should be tested independently of training.

### Artifact-level tests

- log-score conversion produces finite non-negative intensities;
- preprocessing at export matches preprocessing used during fitting;
- state and edge identifiers map to the intended Jact names;
- each fixed-source output is reordered to `state_space.targets(source)`;
- padded outputs are removed;
- both `(1, D)` and `(batch, 1)` duration inputs work;
- scalar and batch-major covariates produce the intended shapes;
- predictions remain stable over the full deployment time-duration domain; and
- saving and loading preserves predictions and topology metadata.

### Jact integration tests

- separate, exit, and grouped assignments cover every transition exactly once;
- wrapper outputs broadcast through `model.solve(...)`;
- the same fitted model works through `model.simulate(...)`;
- `jax.jit` and the required autodiff mode work;
- single- and multi-device results agree within tolerance;
- probability is conserved and remains finite;
- increasing `steps_per_unit` gives acceptable convergence for the largest
  fitted hazards; and
- grouped-model performance is benchmarked against separate exported heads.

### Wrapper tests

The wrapper suite should cover:

- supported broadcast shapes;
- grouped `output_axis` normalization;
- `output_count` mismatch;
- invalid callable, mapping, and axis configuration;
- immutable model state and static apply kwargs; and
- explicit behavior for non-finite raw outputs.

---

## Recommended export boundary

Keep training and Jact integration separate. The fitting package should export
a compact pure-JAX artifact whose public inference function returns
intensity-scale values for either one transition or one source state's valid
outgoing transitions. Then construct thin source/transition adapters with
`feature_fn` and bind them through `jact.wrappers`.

For the proposed joint competing-risks model, the default should be one
fixed-source export per transient state:

```python
predict_exit_rates(
    params,
    static_spec,
    source_id,
    features,
) -> (..., n_valid_exits_for_source)
```

Each adapter must return valid exits in `StateSpace.targets(source)` order and
on the intensity scale. This retains the statistically useful joint outgoing
hazard model while making transition identity, padding removal, feature
semantics, and Jact's solver contract explicit.

Two follow-ups are worth considering in Jact itself:

1. make grouped evaluation a first-class solver block so shared evaluation is
   structurally guaranteed rather than left to compiler optimization; and
2. consider a topology-aware binding helper that accepts a `StateSpace` and
   source, derives `output_count`, and validates/reorders declared targets.

Neither follow-up is required for correctness. The existing wrapper API is
sufficient once the fitted artifact exports finite, non-negative,
correctly-ordered intensity values with Jact-compatible broadcasting.
