# Simulation Performance Implementation and Benchmark Plan

Implementation and benchmark outcomes are recorded in
[`simulation_performance_results.md`](simulation_performance_results.md).

## Summary

Optimize the existing `model.simulate(...)` path without adding a prepared
runner or other public API. Preserve the current seed-to-path mapping,
deterministic equality across device counts, result shapes, validation
behavior, and overflow semantics.

Implement each candidate separately, benchmark it against the current
batch-1000 baseline, and retain only candidates meeting their performance
target without material regressions.

## Implementation Changes

- Cache simulation plans internally by model identity and declared
  initial-state tuple. Keep runtime arrays and covariates dynamic so cached
  data cannot leak between calls or devices.
- Replace repeated per-calendar-cell RNG generation with an eight-jump random
  buffer per trajectory:
  - Generate keys by following the existing chained `split(key, 3)` sequence
    exactly.
  - Index buffered destination and exponential draws by `jump_count`.
  - Refill only after a trajectory consumes its eight-entry chunk.
  - Handle `max_jumps` boundaries at 0, below 8, exactly 8, and above 8.
- Add a structural no-edge fast path:
  - Return empty jump histories without entering the calendar loop.
  - Advance `final_duration` by the horizon and preserve initial states,
    overflow fields, dtypes, and result layout.
  - Do not terminate the general loop early after stochastic absorption
    because doing so would skip the existing full-grid invalid-intensity
    validation.
- Refactor intensity slab assembly to stack canonical edge outputs directly
  instead of initializing a dense zero tensor and scattering each edge into
  it. Preserve scalar, batch, duration-grid, grouped-output, and
  negative/non-finite validation behavior.
- Consolidate host validation transfers:
  - Validate initial masses and durations with one device-to-host summary.
  - Transfer the negative/non-finite intensity summary once after execution.
  - Keep `model.simulate(...)` synchronous when validation errors must be
    raised.
- Prototype internal batch-shape bucketing using padding only when the next
  128-row bucket adds no more than 12.5% rows; combine it with existing
  device-divisibility padding and trim before validation/results. Retain it
  only if the compile-amortization benchmark passes.
- Keep `max_jumps` buffers and result types unchanged; existing measurements
  show no reason to redesign them.
- Document that larger `replicates` values efficiently saturate GPUs while
  preserving the individual batch and public `(individual, replicate, ...)`
  layout.

## Benchmarking

Upgrade `benchmarks/benchmark_simulator.py` to require GPU by default and
report compile-first, median, minimum, p95, trajectory throughput, jump
throughput, overflow count, backend, device, and package versions. Retain
batch 1,000, horizon 30, 12 steps/unit, `max_jumps=64`, one warm-up, and 20
timed runs as the primary baseline.

Add reproducible scenarios and sweeps:

- Standard three-state benchmark.
- No-transition structural fast-path control.
- High-transition multi-jump model exercising RNG-buffer refills.
- Grouped and computationally involved intensities.
- Steps/unit: 1, 2, 4, 8, 12, and 24.
- `max_jumps`: 8, 16, 32, 64, and 128.
- Replicates: 1, 2, 4, 8, and 16.
- Nearby batch sizes around 1,000 to assess optional shape bucketing.
- Compiled memory analysis before and after.

Recorded acceptance targets on the same RTX A4000/JAX environment:

- Primary batch-1000 median: at least 25% faster than the pre-change
  implementation.
- No-transition control: at least 50% faster.
- RNG buffering alone: at least 15% faster on the standard scenario.
- Intensity refactor: at least 5% faster on the standard scenario or 10% on
  involved/grouped scenarios.
- Shape bucketing: at least 15% lower total compile time across nearby batch
  sizes, with no more than 5% steady-state regression.
- Replicate and `max_jumps` sweeps: no configuration more than 10% slower.
- Compile-first time: no more than 10% regression.
- Compiled peak memory: no more than 15% regression.

GPU results remain recorded review targets rather than CI gates. The benchmark
must fail early with a clear diagnostic when JAX, JAXLIB, CUDA plugin, driver,
or cuDNN versions are incompatible.

## Tests and Compatibility

- Capture small pre-change golden outputs and require exact equality for jump
  times, destinations, counts, overflow, and state paths with typed and legacy
  keys.
- Exercise RNG chunk boundaries at 0, 1, 7, 8, 9, 16, and overflow after the
  final slot.
- Preserve bitwise equality across `devices=None`, one device, and supported
  multi-device configurations.
- Test no-edge models with scalar and per-individual initial states and
  nonzero initial durations.
- Confirm invalid intensities are still detected in future cells even when all
  sampled trajectories have become absorbing.
- Retain all intensity broadcasting and grouped-assignment tests.
- Verify plan caching cannot reuse topology or callable assignments across
  different models.
- If shape bucketing is retained, compare padded and exact execution outputs
  bitwise and test trimming, covariates, initial mixtures, and multi-device
  padding.
- Run the complete pytest suite, Ruff, and Pyright after each retained
  implementation stage.

## Assumptions

- `Model.simulate` remains the only user-facing simulation entry point.
- Exact historical RNG paths are preserved.
- Ineffective candidates are reverted.
- The project's broad JAX dependency remains unchanged while the benchmark
  documentation records a known-compatible GPU environment.
