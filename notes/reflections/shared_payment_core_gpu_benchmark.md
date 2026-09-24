# Shared payment core GPU benchmark

Measured on 2026-09-24 at commit `20c22f6` with JAX/JAXLIB 0.4.38 on an
NVIDIA RTX A4000 (16 GB, driver 535.183.01, CUDA driver 12.2). JAX reported
`cuda:0` and the `gpu` backend. All calculations used float32.

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_shared_payment_cores.py \
  --batches 64 256 --steps-per-unit 32 128 --consumers 1 8 \
  --warmups 3 --repeats 15 --output /tmp/jact_shared_payment_gpu_benchmark.json

PYTHONPATH=src .venv/bin/python benchmarks/benchmark_shared_payment_cores.py \
  --batches 256 --steps-per-unit 128 --consumers 2 16 \
  --warmups 3 --repeats 15 --output /tmp/jact_shared_payment_gpu_scaling.json
```

The horizon was four years. The ordinary and shared declarations used the same
probability model, payment cores, weights, and numerical grid. The derived core
used a shared nonlinear duration field. The benchmark compared every output
component with `rtol=3e-5, atol=1e-6` before timing; all 20 comparisons passed.
Each reported execution time is the median of 15 warmed, device-synchronized
calls, with measurement order alternating. Compilation was excluded.

| Batch | Duration cells | Core | Consumers | Ordinary (ms) | Shared (ms) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 128 | cheap | 1 | 2.524 | 2.390 | 1.06x |
| 64 | 128 | derived | 1 | 3.329 | 3.221 | 1.03x |
| 64 | 128 | cheap | 8 | 2.805 | 2.453 | 1.14x |
| 64 | 128 | derived | 8 | 3.598 | 3.276 | 1.10x |
| 64 | 512 | cheap | 1 | 8.349 | 8.335 | 1.00x |
| 64 | 512 | derived | 1 | 11.203 | 11.199 | 1.00x |
| 64 | 512 | cheap | 8 | 9.220 | 8.488 | 1.09x |
| 64 | 512 | derived | 8 | 12.165 | 11.417 | 1.07x |
| 256 | 128 | cheap | 1 | 2.049 | 2.046 | 1.00x |
| 256 | 128 | derived | 1 | 3.331 | 3.360 | 0.99x |
| 256 | 128 | cheap | 8 | 2.726 | 2.368 | 1.15x |
| 256 | 128 | derived | 8 | 4.092 | 3.660 | 1.12x |
| 256 | 512 | cheap | 1 | 10.382 | 9.843 | 1.05x |
| 256 | 512 | derived | 1 | 13.213 | 12.646 | 1.04x |
| 256 | 512 | cheap | 2 | 10.392 | 9.763 | 1.06x |
| 256 | 512 | derived | 2 | 12.924 | 12.466 | 1.04x |
| 256 | 512 | cheap | 8 | 11.170 | 9.996 | 1.12x |
| 256 | 512 | derived | 8 | 14.339 | 12.957 | 1.11x |
| 256 | 512 | cheap | 16 | 12.654 | 9.893 | 1.28x |
| 256 | 512 | derived | 16 | 15.365 | 12.701 | 1.21x |

On this GPU, eight consumers gained 1.07–1.15x across the matched CPU matrix;
one consumer was essentially even. At the larger grid, increasing from two to
16 consumers raised the observed speedup from 1.06x to 1.28x for the cheap core
and from 1.04x to 1.21x for the derived core. The GPU gains are smaller than the
[CPU measurements](shared_payment_core_benchmark.md), where eight consumers
gained 2.77–4.72x. These are end-to-end solver timings for this workload and
device, not isolated payment-kernel timings or a general speedup guarantee.
