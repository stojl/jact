# Simulation Performance Results

Measurements were taken on an NVIDIA RTX A4000 with driver 535.183.01,
JAX/JAXLIB/CUDA plugin 0.4.38, cuDNN 9.1.0.70, Python 3.12.9, and float32
defaults. The primary case used batch 1,000, horizon 30, 12 steps/unit,
`max_jumps=64`, one warm-up, and 20 timed runs.

## Retained changes

| Measurement | Before | Retained implementation | Change |
|---|---:|---:|---:|
| Primary median | 0.047616 s | 0.048070 s | 0.95% slower |
| Primary compile-first | 2.243684 s | 2.395877 s | 6.78% slower |
| Primary peak allocator memory | 6,691,584 B | 6,691,584 B | unchanged |
| No-transition median | 0.021668 s | 0.006920 s | 68.06% faster |
| No-transition compile-first | 1.577862 s | 1.179642 s | 25.24% faster |

The structural no-edge fast path passed its 50% target and was retained.
Identity-scoped simulation-plan caching and consolidated host validation were
also retained. The general stochastic kernel is unchanged after candidate
evaluation, so the primary 25% aggregate target was not reached.

## Rejected candidates

- Two bitwise-equivalent eight-jump RNG-buffer implementations were tested.
  The lower-overhead version had a 0.066324 s primary median, a 39.3%
  regression, and was reverted because it missed the required 15% gain.
- Direct canonical-edge stacking had a 0.048154 s primary median, effectively
  unchanged from baseline and below its 5% target, so it was reverted.
- Padding batch 1,000 to the next 128-row bucket produced a 0.051706 s median,
  a 7.4% steady-state regression. This exceeded the 5% limit, so shape
  bucketing was reverted without using it in production.

All rejected RNG prototypes preserved the historical typed- and legacy-key
outputs exactly at `max_jumps` boundaries 0, 1, 7, 8, 9, and 16 before they
were removed.

## Benchmark coverage

`benchmarks/benchmark_simulator.py` now defaults to the primary GPU case and
reports compile-first, median, minimum, p95, trajectory and jump throughput,
overflow, device padding, allocator memory, backend/device data, driver, and
package versions. It includes:

- standard, no-transition, high-transition, and grouped/involved scenarios;
- steps/unit 1, 2, 4, 8, 12, and 24;
- `max_jumps` 8, 16, 32, 64, and 128;
- replicates 1, 2, 4, 8, and 16;
- nearby batches 896 through 1,056.

The complete scenario and sweep matrix was exercised on the environment above.
At 16 replicates, the primary case processed about 212,000 trajectories/s
while retaining the public `(individual, replicate, ...)` layout.
