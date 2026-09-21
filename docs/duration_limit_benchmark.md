# Duration-limit benchmark

Local CPU measurements of the implemented solver. These are observations,
not portable speed guarantees or accuracy bounds.

## Settings

- Intel Core i5-7300U CPU @ 2.60 GHz; one JAX CPU device.
- Python 3.14.5, JAX 0.10.0, float32.
- Horizons [5, 10, 20]; 12 steps per unit; batch 4; cutoff 2.0 for every enabled limit.
- Two states with transitions in both directions, initially all mass in the
  first state at duration zero; covariate values evenly spaced from zero to one.
- Cheap callables are affine in time, duration, and the covariate. Expensive
  callables add three dense 32-wide tanh layers. Both intensities and payments
  use the selected profile. Cashflows include state rates and transition lumps.
- One compilation/warmup solve per configuration; median of 3 timed solves,
  synchronizing the entire result after each. Timings include public solve
  dispatch and preparation. Other test and benchmark processes were stopped.
- Only initial/final probability diagnostics and terminal cashflows are stored.

Reproduce with:

```bash
JAX_PLATFORMS=cpu .venv/bin/python benchmarks/benchmark_duration_limits.py \
    --horizons 5 10 20 --batch 4 --steps-per-unit 12 --limit 2 --repeats 3 \
    --output /tmp/duration_limits_final.json
```

## Measurements

`intensity`, `payment`, and `probability` enable only that limit; `all` enables
all three. Carry bytes include every numeric state leaf, including point mass
and tail durations, but exclude outputs and temporary arrays. Tail mass is
the final mean across individuals, summed across states. A zero tail for
uncompressed solves means there is no compressed bucket, not that there are
no older cohorts.

Probability error is the maximum absolute state-occupancy error at the
recorded initial/final times, across individuals and states. Cashflow error
is the maximum absolute terminal-total error across individuals. Each is
relative to the exact solve with the same profile, horizon, and inputs.

| Profile | Horizon | Mode | Median ms | Carry bytes | Tail mass | Probability error | Cashflow error |
|---|---:|---|---:|---:|---:|---:|---:|
| cheap | 5 | exact | 3.962 | 1968 | 0.00000 | 0 | 0 |
| cheap | 5 | intensity | 7.756 | 1968 | 0.00000 | 0.0349318 | 0.0861006 |
| cheap | 5 | payment | 5.831 | 1968 | 0.00000 | 0 | 0.341126 |
| cheap | 5 | probability | 4.833 | 912 | 0.25164 | 0.00464141 | 0.053503 |
| cheap | 5 | all | 3.880 | 912 | 0.24703 | 0.0349318 | 0.429625 |
| cheap | 10 | exact | 8.847 | 3888 | 0.00000 | 0 | 0 |
| cheap | 10 | intensity | 7.443 | 3888 | 0.00000 | 0.0511151 | 0.200588 |
| cheap | 10 | payment | 11.600 | 3888 | 0.00000 | 0 | 1.73539 |
| cheap | 10 | probability | 8.595 | 912 | 0.57512 | 0.0312182 | 0.734581 |
| cheap | 10 | all | 7.645 | 912 | 0.50574 | 0.051115 | 2.15389 |
| cheap | 20 | exact | 25.403 | 7728 | 0.00000 | 0 | 0 |
| cheap | 20 | intensity | 16.160 | 7728 | 0.00000 | 0.000791311 | 0.961922 |
| cheap | 20 | payment | 22.671 | 7728 | 0.00000 | 0 | 4.90095 |
| cheap | 20 | probability | 7.369 | 912 | 0.72942 | 0.0175396 | 3.94619 |
| cheap | 20 | all | 9.026 | 912 | 0.68300 | 0.000791311 | 6.04366 |
| expensive | 5 | exact | 59.213 | 1968 | 0.00000 | 0 | 0 |
| expensive | 5 | intensity | 50.679 | 1968 | 0.00000 | 0.0295484 | 0.0872984 |
| expensive | 5 | payment | 48.530 | 1968 | 0.00000 | 0 | 0.326663 |
| expensive | 5 | probability | 32.269 | 912 | 0.27257 | 0.00474879 | 0.0610428 |
| expensive | 5 | all | 33.080 | 912 | 0.26846 | 0.0295484 | 0.416247 |
| expensive | 10 | exact | 183.071 | 3888 | 0.00000 | 0 | 0 |
| expensive | 10 | intensity | 201.556 | 3888 | 0.00000 | 0.0342129 | 0.219938 |
| expensive | 10 | payment | 145.033 | 3888 | 0.00000 | 0 | 1.62011 |
| expensive | 10 | probability | 72.398 | 912 | 0.57495 | 0.0260376 | 0.775293 |
| expensive | 10 | all | 55.146 | 912 | 0.52052 | 0.0342129 | 2.0352 |
| expensive | 20 | exact | 720.579 | 7728 | 0.00000 | 0 | 0 |
| expensive | 20 | intensity | 680.908 | 7728 | 0.00000 | 0.000713468 | 0.581974 |
| expensive | 20 | payment | 485.150 | 7728 | 0.00000 | 0 | 4.51665 |
| expensive | 20 | probability | 109.273 | 912 | 0.69477 | 0.00917959 | 3.87651 |
| expensive | 20 | all | 196.571 | 912 | 0.66665 | 0.000713646 | 5.63973 |

## Interpretation

At horizon 20, the cheap profile's combined solve took 9.026 ms versus 25.403 ms exact (2.81× faster in this run). Its maximum terminal cashflow error was 6.04366.

At horizon 20, the expensive profile's combined solve took 196.571 ms versus 720.579 ms exact (3.67× faster in this run). Its maximum terminal cashflow error was 5.63973.

Function limits alone retain the original carry size. Compression fixes the
regular grid at 25 cells for these horizons, plus one tail, so its carry size
is independent of the simulation horizon. Actual timing also reflects dispatch
overhead, compiler choices, and callable cost; it need not follow the arithmetic
complexity exactly.

The small cutoff intentionally exposes approximation error. These models
continue varying in duration above the cutoff, so neither constant function
tails nor fixed-duration probability tails are exact. Choose cutoffs using
representative model inputs and relevant cashflows. This benchmark does not
include duration events; their omitted continuous payments above the retained
grid are a separate approximation described in the API specification.
