# JACT Event Simulator — Implementation Plan

## 1. Objective

Add a high-throughput, JAX-native semi-Markov path simulator to `jact` that reuses the existing:

- `StateSpace`
- `Model`
- intensity callable contract `fn(t, d, **kwargs)`
- grouped and exit intensity assignments
- reachability reduction
- `InitialDistribution`
- JAX batching and device execution model

The simulator must produce continuous event occurrence times and event-state histories for a portfolio of individuals.

The intensity model is approximated on a rectangular calendar-time and duration grid using midpoint values. Simulation is exact for that discretized piecewise-constant intensity model.

---

## 2. Primary design decisions

### 2.1 Language and execution backend

Implement the first production version entirely in JAX.

Reasons:

- Intensity functions already live in JAX.
- The transition-probability solver already uses the same midpoint discretization.
- A fixed `max_jumps` makes event output compatible with static JAX shapes.
- JIT, `vmap`, `lax.scan`, and `lax.while_loop` are sufficient for the event kernel.
- A Rust backend can remain a future optimization if profiling shows a clear CPU bottleneck.

### 2.2 Public API location

Add simulation as another execution method on the existing `Model`:

```python
result = model.simulate(...)
```

Users define their state space and intensities once, then choose:

```python
model.solve(...)
```

for transition probabilities and expected values, or:

```python
model.simulate(...)
```

for sampled event histories.

### 2.3 Intensity approximation

For calendar-time grid cells

\[
[T_\ell, T_{\ell+1})
\]

and duration grid cells

\[
[D_k, D_{k+1}),
\]

define the discretized intensity as:

\[
\widetilde{\lambda}_{ij}(t,d,x)
=
\lambda_{ij}\left(
\frac{T_\ell + T_{\ell+1}}{2},
\frac{D_k + D_{k+1}}{2},
x
\right).
\]

The intensity is constant inside each time-duration rectangle.

All discontinuities in calendar time must lie on the time grid, and all discontinuities in duration must lie on the duration grid.

Event times remain continuous. They are not rounded to the solver grid.

### 2.4 Event storage

Use fixed-shape event buffers with a user-supplied `max_jumps`.

For:

- `B` individuals
- `R` replicates
- `J = max_jumps`

store arrays with shapes such as:

```text
jump_times       (B, R, J)
jump_durations   (B, R, J)
state_path       (B, R, J + 1)
jump_count       (B, R)
overflow         (B, R)
```

---

## 3. Existing JACT components to reuse

### 3.1 `StateSpace`

Reuse without changing its public responsibility.

It already provides:

- ordered state names
- state-to-index mapping
- allowed transitions
- absorbing-state detection
- outgoing transition queries
- reachability
- model construction
- initial-condition helpers

Recommended addition:

```python
def ordered_transitions(
    self,
    states: Sequence[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    ...
```

This should return transitions ordered by:

1. source-state order
2. target-state order

This gives deterministic edge IDs.

An internal helper is sufficient if a public method is not desirable.

### 3.2 `Model`

Reuse:

- transition assignment validation
- `TransitionInfo`
- `transitions=`
- `exits=`
- `groups=`
- reachable-state reduction
- intensity callable binding

Add:

```python
def simulate(...) -> SimulationResult:
    ...
```

Do not force users to construct another model object.

### 3.3 Intensity protocols

Keep the existing contracts:

```python
def intensity(t, d, **kwargs):
    ...
```

Single-transition output:

```text
broadcastable to (batch, duration)
```

Grouped output:

```text
(transition, ...) where each selected transition is
broadcastable to (batch, duration)
```

No new user-side intensity interface is required.

### 3.4 `InitialDistribution`

Reuse the existing initial-condition semantics.

The simulator must support:

- one common initial state
- per-individual initial states
- scalar or per-individual initial duration
- structurally declared initial-state sets
- component mixtures

For component mixtures, sample the initial state once per trajectory according to the supplied mass vector.

### 3.5 Multi-device support

Mirror the existing `devices` option where practical.

Start with:

- single-device JIT
- CPU and GPU compatibility

Add batch sharding after the single-device implementation is correct and benchmarked.

---

## 4. New internal representation

The current dense solver matrix is appropriate for the probability solver but should not be the main representation for simulation.

The simulator needs:

- deterministic sparse edge IDs
- fast outgoing-edge lookup
- grouped callable preservation
- JAX-static shapes

### 4.1 `SimulationPlan`

Add an internal immutable plan:

```python
from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class SimulationPlan:
    initial_states: tuple[str, ...]
    states: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    edge_sources: jax.Array
    edge_targets: jax.Array

    outgoing_edge_ids: jax.Array
    outgoing_targets: jax.Array
    outgoing_valid: jax.Array

    intensity_blocks: tuple["IntensityBlock", ...]

    n_states: int
    n_edges: int
    max_out_degree: int
```

Suggested array dtypes:

```text
state indices       int32
edge indices        int32
valid masks         bool
times/durations     float32 or float64, following JAX configuration
```

### 4.2 `IntensityBlock`

Preserve grouped callable evaluation:

```python
@dataclass(frozen=True)
class IntensityBlock:
    fn: Any
    edge_ids: tuple[int, ...]
    output_indices: tuple[int | None, ...]
```

Examples:

```text
Single-transition callable
    edge_ids       = (4,)
    output_indices = (None,)

Exit-group callable
    edge_ids       = (1, 2, 3)
    output_indices = (0, 1, 2)

Arbitrary grouped callable
    edge_ids       = (0, 5, 8)
    output_indices = (0, 1, 2)
```

Evaluate each grouped callable once per calendar-time slab and scatter its outputs into canonical edge slots.

Do not use one sliced wrapper call per transition in the simulation path.

### 4.3 Padded sparse adjacency

Use padded outgoing adjacency:

```text
outgoing_edge_ids  (n_states, max_out_degree)
outgoing_targets   (n_states, max_out_degree)
outgoing_valid     (n_states, max_out_degree)
```

Example:

```text
state 0: edges [0, 1, -, -]
state 1: edges [2, 3, 4, -]
state 2: edges [-, -, -, -]
```

This is JAX-friendly because all shapes are static.

If out-degree imbalance becomes a major performance problem, add degree buckets later.

---

## 5. Public API proposal

### 5.1 Basic use

```python
key = jax.random.key(42)

result = model.simulate(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    initial_duration=0.0,
    max_jumps=128,
    replicates=100,
    key=key,
    overflow="return",
    age=ages,
    smoker=smoker,
)
```

### 5.2 Proposed signature

```python
def simulate(
    self,
    initial: str | ArrayLike | InitialDistribution,
    horizon: int,
    steps_per_unit: int,
    initial_duration: ArrayLike = 0.0,
    *,
    max_jumps: int,
    replicates: int = 1,
    key: jax.Array,
    overflow: Literal["return", "raise"] = "return",
    devices: int | Sequence[Any] | None = None,
    **kwargs: Any,
) -> SimulationResult:
    ...
```

### 5.3 Grid semantics

Version 1 should mirror `solve`:

```text
time step      = 1 / steps_per_unit
duration step  = 1 / steps_per_unit
time horizon   = horizon
```

Both grids use the same step size.

A later extension can support distinct grids:

```python
model.simulate(
    time_grid=...,
    duration_grid=...,
    ...
)
```

The first release should avoid that additional API surface unless there is an immediate use case.

### 5.4 Result type

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationResult:
    states: tuple[str, ...]

    jump_times: jax.Array
    jump_durations: jax.Array
    state_path: jax.Array

    jump_count: jax.Array
    overflow: jax.Array
    truncated_at_time: jax.Array

    final_state: jax.Array
    final_duration: jax.Array
```

Interpretation:

```python
state_path[b, r, 0]       # initial state
state_path[b, r, j]       # source state of jump j
state_path[b, r, j + 1]   # destination state of jump j

jump_times[b, r, j]       # elapsed calendar time of jump j
jump_durations[b, r, j]   # state duration immediately before jump j
```

The valid mask is derived as:

```python
valid = (
    jnp.arange(max_jumps)
    < result.jump_count[..., None]
)
```

### 5.5 Convenience methods

Outside the JIT kernel, provide:

```python
result.valid_mask()
result.to_pandas()
result.path(individual=12, replicate=3)
```

`to_pandas()` should return columns:

```text
individual
replicate
jump_number
jump_time
jump_duration
from_state
to_state
```

Avoid making pandas a mandatory dependency. Put it behind an optional import.

---

## 6. Intensity slab evaluation

### 6.1 Match the existing solver discretization

For each calendar-time cell:

```python
t_mid = t_left + 0.5 * step_size
```

Define all duration-cell midpoints:

```python
duration_mid = (
    jnp.arange(n_duration_cells) + 0.5
) * step_size
```

Call each intensity block with:

```python
d = duration_mid[None, :]
```

and covariates broadcast in the existing way.

The target logical shape for one transition is:

```text
(batch, duration_cells)
```

The complete rate slab is:

```text
(edge, batch, duration_cell)
```

### 6.2 Suggested evaluator

```python
def evaluate_intensity_slab(
    plan: SimulationPlan,
    t_mid: jax.Array,
    duration_mid: jax.Array,
    kwargs: dict[str, jax.Array],
    batch_size: int,
) -> jax.Array:
    rates = jnp.zeros(
        (plan.n_edges, batch_size, duration_mid.shape[-1]),
        dtype=duration_mid.dtype,
    )

    d = duration_mid[None, :]

    for block in plan.intensity_blocks:
        raw = jnp.asarray(block.fn(t_mid, d, **kwargs))

        if block.output_indices == (None,):
            value = broadcast_grid_output(
                raw,
                (batch_size, duration_mid.shape[-1]),
            )
            rates = rates.at[block.edge_ids[0]].set(value)
            continue

        normalized = normalize_grouped_output(raw)

        for edge_id, output_index in zip(
            block.edge_ids,
            block.output_indices,
            strict=True,
        ):
            value = broadcast_grid_output(
                normalized[output_index],
                (batch_size, duration_mid.shape[-1]),
            )
            rates = rates.at[edge_id].set(value)

    return jnp.maximum(rates, 0.0)
```

The Python loop over intensity blocks is static at trace time. The resulting JAX computation remains compiled.

### 6.3 Validation

At model preparation or first trace:

- verify every edge is assigned exactly once
- verify outputs broadcast correctly
- verify finite values
- reject materially negative values
- optionally clamp tiny negatives caused by numerical noise
- verify all required covariate batch dimensions agree

Suggested negative policy:

```python
negative_tolerance=1e-12
```

Values below `-negative_tolerance` should raise.

Values in `[-negative_tolerance, 0)` may be clamped to zero.

---

## 7. Event simulation kernel

## 7.1 State carried per trajectory

Flatten individual and replicate dimensions internally:

```text
trajectory_count = batch * replicates
```

Carry arrays of shape `(trajectory_count,)`:

```text
current_state
current_time
current_duration
remaining_hazard
jump_count
overflow
finished
random_key
```

Carry event buffers:

```text
jump_times       (trajectory_count, max_jumps)
jump_durations   (trajectory_count, max_jumps)
state_path       (trajectory_count, max_jumps + 1)
```

### 7.2 RNG state

For each trajectory:

1. derive a deterministic key from:
   - global key
   - individual index
   - replicate index
2. sample an initial exponential hazard threshold
3. split keys for:
   - new exponential thresholds
   - destination selection

Results should not depend on portfolio batching.

### 7.3 Outer loop

Use `jax.lax.scan` over calendar-time cells.

Each scan step:

1. evaluate the full duration-intensity slab at the calendar midpoint
2. simulate all active trajectories until they reach the cell endpoint
3. carry trajectory state into the next calendar cell

Conceptual structure:

```python
final_carry, _ = jax.lax.scan(
    simulate_calendar_cell,
    initial_carry,
    calendar_cells,
)
```

### 7.4 Inner loop

Use `jax.lax.while_loop`.

The loop continues while at least one non-finished trajectory has not reached the end of the current calendar cell.

At every iteration:

1. determine each trajectory's duration-cell index
2. gather outgoing rates for its current state
3. compute the total rate
4. compute distance to the next duration boundary or calendar boundary
5. either:
   - cross a boundary, or
   - emit a jump
6. update buffers and state

### 7.5 Gather current rates

Given:

```text
rate_slab              (edge, trajectory, duration_cell)
outgoing_edge_ids       (state, max_out_degree)
outgoing_targets        (state, max_out_degree)
outgoing_valid          (state, max_out_degree)
```

gather:

```text
outgoing_rates          (trajectory, max_out_degree)
```

Use safe edge IDs for padded slots and mask invalid entries to zero.

### 7.6 Boundary distance

For each active trajectory:

```python
duration_idx = jnp.floor(duration / step_size).astype(jnp.int32)
duration_right = (duration_idx + 1) * step_size

dt_duration = duration_right - duration
dt_calendar = t_right - current_time

dt_boundary = jnp.minimum(dt_duration, dt_calendar)
```

Handle numerical near-boundary values carefully to prevent zero-length loops.

Recommended approach:

- snap indices using a small dtype-aware tolerance
- after crossing a boundary, set time or duration exactly to the computed boundary
- ensure every loop iteration makes positive progress or emits a jump

### 7.7 Hazard-threshold algorithm

Let:

```text
q = sum(outgoing_rates)
E = remaining_hazard
r = dt_boundary
```

If `q == 0`:

```text
advance by r
leave E unchanged
```

If:

\[
E > q r,
\]

then:

```text
advance by r
E <- E - q*r
```

Otherwise the event occurs after:

\[
\Delta = E/q.
\]

Record:

```text
jump_time      = current_time + Delta
jump_duration  = current_duration + Delta
```

Select the destination proportionally to outgoing rates, reset duration to zero, and sample a new exponential threshold.

### 7.8 Destination selection

For each firing trajectory:

```python
u = jax.random.uniform(key)
threshold = u * total_rate

cumulative = jnp.cumsum(outgoing_rates, axis=-1)
slot = jnp.argmax(cumulative >= threshold[:, None], axis=-1)

destination = gather(outgoing_targets, slot)
```

Only apply selected values for firing trajectories.

### 7.9 Multiple jumps in one calendar cell

The inner loop must allow an arbitrary number of jumps up to `max_jumps`.

After a jump:

```text
current_time      remains at jump time
current_duration  becomes 0
current_state     becomes destination
remaining_hazard  becomes a newly sampled Exp(1)
```

Continue until the trajectory reaches the calendar-cell endpoint or stops.

---

## 8. Overflow behavior

Overflow must never be silent.

When a trajectory needs another event but:

```text
jump_count == max_jumps
```

set:

```text
overflow = True
truncated_at_time = current_time
finished = True
```

Do not write beyond the buffer.

### 8.1 Public policies

```python
overflow="return"
```

Return flags and all recorded events.

```python
overflow="raise"
```

Run the compiled kernel, inspect overflow on the host, and raise an informative exception.

Example message:

```text
17 of 100,000 trajectories exceeded max_jumps=128.
Increase max_jumps or inspect the high-transition trajectories.
```

Do not raise from inside the JIT kernel.

---

## 9. Initial-condition handling

### 9.1 Single state

```python
initial="healthy"
```

Broadcast the state index across individuals and replicates.

### 9.2 Per-individual state

Reuse:

```python
state_space.initial_per_individual(...)
```

Flatten across replicates after canonicalization.

### 9.3 Initial duration

Support scalar or `(batch,)` duration.

Replicate per-individual values across simulation replicates.

The duration grid must cover:

\[
\max(\text{initial duration}) + \text{horizon}.
\]

Version 1 options:

1. allocate enough duration cells automatically
2. reject values beyond a configured maximum

Preferred default:

```text
n_duration_cells =
ceil((max_initial_duration + horizon) * steps_per_unit)
```

This keeps the approximation well-defined without a hidden tail rule.

### 9.4 Component mixtures

For an `InitialDistribution` with masses over several structural states:

1. canonicalize masses and durations
2. sample one component per trajectory
3. select that component's initial duration
4. use the structurally declared state set for topology reduction

---

## 10. Sparse topology reduction

Compile the simulation plan from the same declared initial-state semantics as `solve`.

Steps:

1. resolve structurally active initial states
2. compute reachable states
3. preserve reduced-state order
4. include only edges whose source and target are reachable
5. remap full state indices to reduced state indices
6. build deterministic reduced edge IDs
7. build padded adjacency
8. build grouped intensity blocks

The simulation result's `states` field should use reduced-state order, matching the rest of `jact`.

---

## 11. File-level implementation plan

### 11.1 New files

```text
src/jact/simulation.py
```

Contains:

- `SimulationPlan`
- `IntensityBlock`
- input preparation
- topology compilation
- slab evaluator
- JAX event kernel
- public module-level `simulate()`

```text
src/jact/simulation_result.py
```

Contains:

- `SimulationResult`
- PyTree registration
- valid-mask helper
- path extraction helper
- optional pandas conversion

### 11.2 Existing files to modify

```text
src/jact/model.py
```

Add:

- `Model.simulate(...)`
- import and delegation to `jact.simulation.simulate`

Potentially expose an internal read-only iterator over transition assignments so simulation code does not depend heavily on private dictionaries.

Suggested internal method:

```python
def _simulation_assignments(
    self,
    reachable_states: Sequence[str],
) -> tuple[TransitionInfo, ...]:
    ...
```

```text
src/jact/state_space.py
```

Optional:

- deterministic ordered-transition helper

```text
src/jact/__init__.py
```

Export:

- `SimulationResult`

Keep internal plan types private.

```text
src/jact/typing.py
```

No required public changes.

Potentially add result-related type aliases only if useful.

### 11.3 Shared helpers

The existing solver already has useful broadcasting helpers.

Consider extracting:

```text
_broadcast_grid_output
_broadcast_vector_output
_broadcast_point_output
```

into:

```text
src/jact/_broadcast.py
```

Both the probability solver and simulator can use them.

Do this only if it reduces duplication without destabilizing the existing solver.

---

## 12. Testing plan

## 12.1 Unit tests: topology and compilation

Test:

- deterministic edge ordering
- absorbing states
- multiple reachable initial states
- unreachable-state removal
- padded adjacency values
- grouped-callable block construction
- exit-callable block construction
- single-transition blocks
- duplicate and missing assignment behavior remains unchanged

### 12.2 Unit tests: intensity evaluation

Test callable outputs that are:

- scalar
- duration-only
- batch-only
- full `(batch, duration)`
- grouped on leading axis
- produced by fitted-model wrappers

Verify:

- broadcasting
- grouping
- non-negativity
- covariate propagation
- midpoint evaluation
- discontinuities aligned with the grid

### 12.3 Exact distribution tests

#### Constant one-way hazard

Model:

```text
A -> B with rate lambda
B absorbing
```

Compare simulated jump-time distribution with:

\[
P(T \le t)=1-e^{-\lambda t}.
\]

Compare:

- empirical event probability
- empirical mean event time conditional on event before horizon
- final-state probability

#### Competing constant hazards

Model:

```text
A -> B at lambda_1
A -> C at lambda_2
```

Verify:

\[
P(A \to B)=\frac{\lambda_1}{\lambda_1+\lambda_2}
\]

and exponential waiting time with total rate:

\[
\lambda_1+\lambda_2.
\]

#### Zero-rate interval followed by positive rate

Model:

\[
\lambda(d)=
\begin{cases}
0, & d<1,\\
0.3, & d\ge 1.
\end{cases}
\]

Verify:

- no jumps before duration 1
- post-1 waiting time is exponential with rate 0.3
- event times are continuous rather than grid-snapped

#### Calendar-time discontinuity

Verify no events occur before a configured time threshold when the intensity is zero, then compare the post-threshold distribution.

### 12.4 Semi-Markov tests

Use a model with:

- recovery rate depending on disability duration
- mortality rate depending on calendar time
- duration reset after recovery or disability onset

Verify path invariants:

- duration increases at unit speed between events
- duration resets to zero after every jump
- event source equals previous state
- no undeclared transitions occur
- event times are strictly increasing
- all event times lie within the horizon

### 12.5 Comparison with `solve`

For large Monte Carlo samples, compare empirical state probabilities at solver record times with:

```python
model.solve(...)
```

Use confidence-interval-aware tolerances.

This is the strongest end-to-end test because both methods consume the same state space, intensities, grid, and covariates.

### 12.6 Reproducibility tests

Verify:

- same key gives identical results
- different keys change results
- results do not change when portfolio batching changes
- replicate ordering is deterministic
- JIT and non-JIT paths agree

### 12.7 Overflow tests

Verify:

- `max_jumps=0`
- exact filling of the buffer
- overflow flags
- truncation time
- `overflow="raise"`
- no out-of-bounds writes

### 12.8 Initial-distribution tests

Test:

- one initial state
- per-individual state
- per-individual duration
- component mixture sampling
- declared zero-mass states still affecting topology reduction
- multiple replicates

### 12.9 Device tests

Where CI hardware allows:

- CPU JIT
- GPU JIT
- single-device equivalence

Multi-device tests can be added after the implementation exists.

---

## 13. Benchmark plan

Benchmark compilation and execution separately.

Always call:

```python
result.jump_times.block_until_ready()
```

before stopping execution timers.

### 13.1 Benchmark dimensions

Portfolio sizes:

```text
1,000
10,000
100,000
1,000,000
```

Replicates:

```text
1
10
100
```

State counts:

```text
3
10
50
200
```

Out-degree patterns:

```text
uniform sparse
highly imbalanced sparse
moderately dense
```

Grid resolution:

```text
12 steps/year
52 steps/year
365 steps/year
```

Maximum jumps:

```text
16
64
128
512
```

Intensity costs:

```text
simple parametric
GLM
small neural network
grouped neural network
```

### 13.2 Metrics

Measure:

- JIT compilation time
- simulation wall time
- trajectories per second
- jumps per second
- peak memory
- time spent evaluating intensities
- time spent in event traversal
- effect of `max_jumps`
- effect of grouped versus per-transition intensities
- CPU thread scaling
- GPU throughput

### 13.3 Benchmark comparison points

Compare:

1. pointwise intensity evaluation in the inner loop
2. full duration slab per calendar cell
3. grouped callable preservation
4. sliced grouped callables
5. different padded adjacency widths
6. optional state-degree bucketing

The initial implementation should favor clarity. Optimize only after profiling.

---

## 14. Delivery phases

## Phase 1 — Core single-device simulator

Deliver:

- `SimulationResult`
- deterministic sparse simulation plan
- single and grouped intensity slab evaluation
- `Model.simulate`
- single-device JIT
- fixed event buffers
- continuous jump times
- overflow flags
- single-state and per-individual initial conditions

Acceptance criteria:

- constant-hazard tests pass
- discontinuity tests pass
- path invariants pass
- empirical probabilities agree with `solve`

## Phase 2 — Full initial distributions and ergonomics

Deliver:

- component-mixture initial sampling
- convenience path accessors
- optional pandas conversion
- improved error messages
- documentation and examples

Acceptance criteria:

- all existing initial forms are supported
- example notebook demonstrates probability and path simulation from one model

## Phase 3 — Performance tuning

Deliver based on profiling:

- optimized grouped slab evaluation
- portfolio batching
- state-degree bucketing if needed
- memory tuning
- CPU and GPU benchmark suite

Acceptance criteria:

- benchmark report
- documented recommended batch sizes and `max_jumps`
- no regression in `solve`

## Phase 4 — Multi-device execution

Deliver:

- local-device batch sharding
- deterministic key derivation across devices
- output concatenation
- multi-device tests

## Phase 5 — Optional advanced features

Potential additions:

- distinct time and duration grids
- transition counts without event histories
- cashflow evaluation along simulated paths
- user-selectable event payloads
- native CPU backend if benchmarks justify it
- compact host-side ragged conversion
- path-weighting or likelihood outputs

---

## 15. Suggested documentation example

```python
import jax
import jax.numpy as jnp
import jact


state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "healthy"),
        ("disabled", "dead"),
    ],
)


def incidence(t, d, *, age):
    del d
    return 0.01 * jnp.exp(0.04 * (age[:, None] + t - 50.0))


def healthy_mortality(t, d, *, age):
    del d
    attained_age = age[:, None] + t
    return 0.001 * jnp.exp(0.08 * (attained_age - 50.0))


def recovery(t, d, *, age):
    del t, age
    return jnp.where(d < 1.0, 0.8, 0.2)


def disabled_mortality(t, d, *, age):
    attained_age = age[:, None] + t
    return (
        0.003
        * jnp.exp(0.08 * (attained_age - 50.0))
        * (1.0 + 0.5 * d)
    )


model = state_space.build(
    transitions={
        ("healthy", "disabled"): incidence,
        ("healthy", "dead"): healthy_mortality,
        ("disabled", "healthy"): recovery,
        ("disabled", "dead"): disabled_mortality,
    }
)

ages = jnp.linspace(30.0, 80.0, 100_000)

result = model.simulate(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    max_jumps=64,
    replicates=10,
    key=jax.random.key(42),
    age=ages,
)

valid = (
    jnp.arange(result.jump_times.shape[-1])
    < result.jump_count[..., None]
)
```

---

## 16. Definition of done

The first stable simulator release is complete when:

- the same `StateSpace` and `Model` can be used for `solve` and `simulate`
- all existing intensity assignment styles work
- midpoint discretization is identical in meaning across both methods
- event times are continuous
- event states and durations are recorded
- per-individual covariates work
- fixed `max_jumps` buffers work under JIT
- overflow is explicit
- topology reduction matches `solve`
- empirical simulation results agree with transition probabilities
- CPU and GPU benchmarks are published
- public examples and API documentation are complete

---

## 17. Immediate implementation sequence

A practical coding order is:

1. Add `SimulationResult`.
2. Add deterministic reduced edge ordering.
3. Compile `SimulationPlan`.
4. Implement grouped intensity slab evaluation.
5. Implement one-trajectory constant-cell transition logic.
6. Vectorize the transition logic over trajectories.
7. Add the inner `lax.while_loop`.
8. Add the outer calendar-cell `lax.scan`.
9. Add event-buffer writes and overflow.
10. Add `Model.simulate`.
11. Add constant-rate and discontinuity tests.
12. Add comparison tests against `Model.solve`.
13. Add initial-distribution support.
14. Add benchmarks.
15. Add documentation and examples.

This order isolates correctness before performance tuning and keeps the existing probability solver stable.
