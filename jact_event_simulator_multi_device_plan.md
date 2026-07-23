# JACT Event Simulator — Multi-Device Implementation Plan

## 1. Objective

Add deterministic, local multi-device execution to `Model.simulate(...)` while
preserving the single-device API and all current simulation semantics.

The implementation must:

- shard portfolio individuals across selected local JAX devices
- evaluate intensity callables with their existing batch contract
- preserve exact trajectory RNG streams across device counts
- return the same public `(individual, replicate, ...)` array layout
- preserve component-mixture, per-individual, overflow, and validation behavior
- support non-divisible portfolio sizes through safe padding
- avoid changing `Model.solve(...)` behavior

Multi-host execution is out of scope for this phase.

---

## 2. Public API and semantics

Keep the current signature:

```python
result = model.simulate(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    max_jumps=64,
    replicates=10,
    key=jax.random.key(42),
    devices=4,
    age=ages,
)
```

Supported `devices` forms:

```text
None
    Use the existing single-device JIT path.

1
    Explicitly select the first local device and use the single-device path.

N > 1
    Select the first N local devices and use batch-sharded execution.

Sequence[jax.Device]
    Use the supplied local devices in the supplied order.
```

Validation should match `Model.solve(...)`:

- reject booleans
- reject non-positive integers
- reject empty sequences
- reject requests exceeding available local devices
- reject non-local or duplicate devices with a clear error

The result must not expose a device axis. It retains the current shapes:

```text
jump_times         (batch, replicates, max_jumps)
jump_durations     (batch, replicates, max_jumps)
state_path         (batch, replicates, max_jumps + 1)
jump_count         (batch, replicates)
overflow           (batch, replicates)
truncated_at_time  (batch, replicates)
final_state        (batch, replicates)
final_duration     (batch, replicates)
```

---

## 3. Core design decision: shard individuals, not trajectories

Shard the canonical individual batch before replicate expansion.

For:

```text
B = individual count
R = replicate count
D = selected device count
```

prepare device-local inputs as:

```text
masses             (D, local_batch, component)
durations          (D, local_batch, component)
batch covariates   (D, local_batch, ...)
trajectory keys    (D, local_batch, R, key...)
```

Inside each mapped device:

1. flatten `(local_batch, R)` to local trajectories
2. repeat canonical masses and durations across `R`
3. construct local rate-gather indices from `0..local_batch-1`
4. call the existing event kernel
5. reshape output back to `(local_batch, R, ...)`

Reasons:

- intensity functions continue to see an individual batch, not a
  replicate-expanded batch
- grouped slab evaluation remains once per local individual batch
- covariate semantics remain identical to `solve(...)`
- event buffers and rate slabs are distributed across devices
- public individual/replicate ordering is straightforward to reconstruct

Do not shard duration cells, edges, states, or replicates in the first
multi-device implementation.

---

## 4. Deterministic RNG across device counts

Current single-device trajectory keys are derived from:

```python
jax.random.fold_in(
    jax.random.fold_in(global_key, individual_index),
    replicate_index,
)
```

Preserve that exact rule.

Before sharding, construct keys from global, unpadded individual indices:

```text
trajectory_keys  (padded_batch, replicates, key...)
```

Then reshape them to:

```text
(device, local_batch, replicates, key...)
```

The mapped kernel must receive these prepared keys. It must not derive keys
from device index or device-local individual index.

Acceptance requirement:

```text
devices=None
devices=1
devices=2
devices=all_local_devices
```

must produce bitwise-identical results for the same model, inputs, and key,
apart from device-placement metadata.

This includes:

- initial component selection
- initial exponential threshold
- every destination draw
- every subsequent exponential threshold

---

## 5. Padding strategy

When `batch % device_count != 0`, pad the individual axis to:

```python
padded_batch = ceil(batch / device_count) * device_count
```

Do not zero-pad simulation inputs. Repeat the final real individual instead:

```text
component masses
component durations
every batch covariate leaf
```

Reasons:

- zero-valued covariates may be outside a fitted model's valid domain
- zero masses are invalid for component sampling
- repeated valid input avoids padding-only NaNs, infinities, or validation
  failures
- padded outputs are discarded, so their sampled paths are irrelevant

Generate unique global indices and keys for padded rows, but never include
padded rows in:

- public outputs
- overflow counts
- error messages
- statistical summaries

Reject an empty portfolio before padding; there is no valid final row to repeat.

---

## 6. Kernel refactor

Split the current kernel into:

```python
def _event_kernel_impl(...):
    ...

_event_kernel = jax.jit(
    _event_kernel_impl,
    static_argnames=(...),
)
```

Add a mapped wrapper:

```python
def _event_kernel_pmapped_wrapper(
    plan,
    local_masses,
    local_durations,
    initial_state_options,
    local_trajectory_keys,
    calendar_left,
    calendar_right,
    duration_mid,
    batch_kwargs,
    scalar_kwargs,
    *,
    replicates,
    max_jumps,
    negative_tolerance,
):
    ...
```

The wrapper should:

1. merge scalar and local batch kwargs
2. repeat masses and durations over replicates
3. flatten prepared trajectory keys
4. create device-local individual indices for intensity slab gathering
5. invoke `_event_kernel_impl`
6. reshape trajectory-major fields back to local batch-major fields

Keep `SimulationPlan`, grids, initial-state options, scalar kwargs, and static
configuration replicated across devices.

Suggested mapped axes:

```text
plan                     None
local_masses             0
local_durations          0
initial_state_options    None
local_trajectory_keys    0
calendar_left            None
calendar_right           None
duration_mid             None
batch_kwargs             0 per leaf
scalar_kwargs            None per leaf
```

Static mapped arguments:

```text
replicates
max_jumps
negative_tolerance
```

Use one module-level `pmap` for all local devices and a cached factory for an
explicit device tuple, following the existing solver pattern.

---

## 7. Input preparation and sharding helpers

Add simulation-specific helpers in `src/jact/simulation.py` initially:

```python
def _split_scalar_and_batch_kwargs(...)
def _pad_batch_array_by_repetition(...)
def _shard_simulation_batch_array(...)
def _shard_simulation_batch_tree(...)
def _unshard_simulation_result(...)
def _resolve_devices(...)
```

Avoid immediately extracting the solver's existing helpers. Its padding and
time-major unsharding semantics differ from the simulator's requirements.

After both paths are stable, a follow-up cleanup may extract only genuinely
shared helpers such as local-device resolution.

The simulation sharding helper should return:

```text
sharded value
original batch size
padded batch size
local batch size
```

All sharded leaves must have the same original batch size.

---

## 8. Result reconstruction

Have the mapped wrapper return each event field as:

```text
(device, local_batch, replicates, ...)
```

Reconstruct public arrays with:

```python
merged = value.reshape(
    padded_batch,
    replicates,
    *value.shape[3:],
)
public = merged[:original_batch_size]
```

Scalar validation flags returned per device should be reduced with `any`.

Build `SimulationResult` only after:

- removing padding
- reducing validation flags
- checking the requested overflow policy

For `overflow="raise"`, count only unpadded public trajectories.

---

## 9. Validation and error behavior

Preserve current validation:

- finite intensities
- negative-intensity tolerance
- positive mass totals
- finite and non-negative durations
- covariate batch agreement
- explicit overflow behavior

Device-local invalid intensity flags should be aggregated on the host:

```python
invalid_negative = any(device_invalid_negative)
invalid_nonfinite = any(device_invalid_nonfinite)
```

Because repeated padding uses real inputs, a padding row should not introduce a
new validation failure. Still discard padding before overflow counting and
result construction.

If a device execution fails, allow JAX to report the mapped device failure;
do not silently retry on one device.

---

## 10. File-level changes

### `src/jact/simulation.py`

Add:

- device resolution
- repeat-padding helpers
- scalar/batch kwarg splitting
- kernel implementation/wrapper separation
- all-local-device `pmap`
- explicit-device mapped factory
- mapped result reconstruction
- dispatch in `_run_event_kernel(...)`

Replace the current `NotImplementedError` for more than one device.

### `tests/test_simulation.py`

Keep ordinary single-device tests and add validation tests that do not require
multiple hardware devices.

### `tests/test_simulation_multi_device.py`

Run in an isolated subprocess with:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4
JAX_PLATFORMS=cpu
```

This environment must be established before importing JAX.

### `benchmarks/benchmark_simulator.py`

Add:

```text
--devices
```

Report:

- compile plus first-run time
- steady-state execution time
- trajectories per second
- jumps per second
- per-device local batch size
- padding count

### `docs/api_spec.md`

Replace the current single-device limitation with exact device-selection,
padding, determinism, and local-only semantics.

### `README.md`

Add a short multi-device example after the basic simulation example.

---

## 11. Test plan

### 11.1 Device selection

Test:

- `devices=None`
- `devices=1`
- `devices=2`
- explicit two-device sequence
- all local devices
- zero, negative, boolean, empty, duplicate, unavailable, and non-local
  selections

### 11.2 Exact reproducibility

For one model/input/key, assert every result field is equal across:

```text
single device
two devices
four devices
```

Use `equal_nan=True` for fixed jump buffers.

Cover:

- typed keys
- legacy keys
- component mixtures
- per-individual initial states and durations
- multiple replicates
- multiple jumps in one calendar cell
- grouped and exit assignments

### 11.3 Padding

Use batch sizes:

```text
1, 2, 3, 5, 7, 10
```

against device counts:

```text
2, 3, 4
```

Verify:

- exact public shape
- original order retained
- no padding rows returned
- overflow counts exclude padding
- padding does not cause intensity-validation failures

Include a fitted-style intensity that would fail on zero-padded covariates to
prove repetition padding is used.

### 11.4 Statistical and path invariants

Repeat current:

- constant one-way hazard
- competing hazards
- duration discontinuity
- calendar discontinuity
- solve comparison
- duration-reset/path invariant

Run these primarily as exact single-vs-multi comparisons; the existing
single-device statistical tests remain the distribution reference.

### 11.5 Hardware coverage

CI:

- forced multi-device CPU subprocess

Where available:

- two-GPU smoke test
- selected GPU subset
- CPU/GPU statistical agreement

Exact bitwise CPU/GPU equality is not required because floating-point
implementations may differ. Exact equality is required across device counts on
the same backend.

---

## 12. Benchmark plan

Benchmark device counts:

```text
1, 2, 4, 8
```

Portfolio sizes:

```text
10,000
100,000
1,000,000
```

Replicates:

```text
1, 10, 100
```

Record:

- compilation time
- steady-state execution time
- trajectories per second
- jumps per second
- peak device memory
- scaling efficiency relative to one device
- padding overhead for non-divisible batches

Always block on:

```python
result.jump_times.block_until_ready()
```

Do not claim speedup until intensity-slab and event-buffer memory are both
distributed as expected in profiler traces.

---

## 13. Implementation sequence

1. Refactor `_event_kernel` into implementation plus single-device wrapper.
2. Add repeat-padding and batch-tree sharding helpers.
3. Split scalar and batch covariates.
4. Precompute global individual/replicate trajectory keys.
5. Add the device-local mapped wrapper.
6. Add all-local-device `pmap`.
7. Add cached explicit-device `pmap`.
8. Reshape and unpad mapped output.
9. Aggregate validation and overflow flags.
10. Replace multi-device `NotImplementedError` with dispatch.
11. Add forced multi-device CPU equivalence tests.
12. Add padding and RNG invariance tests.
13. Update API documentation and README.
14. Extend the benchmark harness.
15. Run CPU and available GPU profiling.

---

## 14. Definition of done

Multi-device simulation is complete when:

- `Model.simulate(..., devices=N)` runs on selected local devices
- all intensity assignment styles work under mapped execution
- output shapes and ordering match the single-device path
- same-backend results are exactly invariant to device count
- non-divisible batches are safely padded and correctly trimmed
- component mixtures and per-individual starts remain deterministic
- overflow and invalid-intensity reporting exclude padding
- forced multi-device CPU tests pass in CI
- available GPU smoke tests pass
- the benchmark harness reports device-count scaling
- `Model.solve(...)` behavior and the single-device simulator do not regress

