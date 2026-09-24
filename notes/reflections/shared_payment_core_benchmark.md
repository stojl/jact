# Shared payment core benchmark

Measured on 2026-09-24 with JAX 0.10.0 on `cpu:0` (float32).

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_shared_payment_cores.py \
  --batches 64 256 --steps-per-unit 32 128 --consumers 1 8 \
  --warmups 3 --repeats 15 --output /tmp/jact_shared_payment_benchmark.json
```

The horizon is four years. The benchmark computes terminal outputs for each
payment separately, with a different per-person, time-varying weight. Both
versions use the same probability model and numerical grid. The derived core
uses a nonlinear duration field already shared and hoisted by the existing
field runtime, so the comparison isolates the additional benefit of sharing
payment contributions.

All output comparisons passed before timing. The table reports median warmed
execution times with device synchronization. Ordinary/shared measurement order
alternates; compilation and lowering are excluded and reported separately by
the script. The full test suite was run after these measurements.

| Batch | Duration cells | Core | Consumers | Ordinary (ms) | Shared (ms) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 128 | cheap | 1 | 6.41 | 6.72 | 0.95x |
| 64 | 128 | derived | 1 | 9.57 | 10.48 | 0.91x |
| 64 | 128 | cheap | 8 | 31.50 | 6.68 | 4.72x |
| 64 | 128 | derived | 8 | 33.76 | 9.04 | 3.74x |
| 64 | 512 | cheap | 1 | 132.51 | 132.29 | 1.00x |
| 64 | 512 | derived | 1 | 157.94 | 157.65 | 1.00x |
| 64 | 512 | cheap | 8 | 580.53 | 147.66 | 3.93x |
| 64 | 512 | derived | 8 | 641.94 | 154.03 | 4.17x |
| 256 | 128 | cheap | 1 | 30.09 | 30.24 | 1.00x |
| 256 | 128 | derived | 1 | 40.07 | 39.20 | 1.02x |
| 256 | 128 | cheap | 8 | 144.02 | 35.37 | 4.07x |
| 256 | 128 | derived | 8 | 162.77 | 44.44 | 3.66x |
| 256 | 512 | cheap | 1 | 832.66 | 1035.36 | 0.80x |
| 256 | 512 | derived | 1 | 782.67 | 778.23 | 1.01x |
| 256 | 512 | cheap | 8 | 2298.67 | 830.62 | 2.77x |
| 256 | 512 | derived | 8 | 2362.84 | 774.16 | 3.05x |

Eight consumers were 2.77–4.72x faster with shared cores across these cases.
One consumer showed no consistent benefit: speedups ranged from 0.80x to 1.02x,
including a 24% increase in execution time in the largest cheap-core case.
Moving multiplication outside the reduction changes the compiled computation
as well as floating-point association; sharing is most useful when multiple
consumers actually reuse the contribution. These CPU measurements are not a
GPU performance claim or a timing guarantee.

Tests separately inspect traced programs to verify that repeated named
references emit a single base contribution calculation for each compatible
attachment context. Wall-clock timing is not used as a CI assertion.
