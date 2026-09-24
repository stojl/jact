# Shared payment core CPU benchmark on the GPU test machine

Measured on 2026-09-24 at commit `20c22f6` with JAX/JAXLIB 0.4.38 and
`JAX_PLATFORMS=cpu`. The machine exposes eight virtual cores of an Intel Xeon
Gold 5315Y CPU at 3.20 GHz under Xen. JAX reported `TFRT_CPU_0`. This uses the same
code, virtual environment, matrix, and timing procedure as the
[GPU benchmark](shared_payment_core_gpu_benchmark.md).

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python \
  benchmarks/benchmark_shared_payment_cores.py \
  --batches 64 256 --steps-per-unit 32 128 --consumers 1 8 \
  --warmups 3 --repeats 15 --output /tmp/jact_shared_payment_cpu_current.json

JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python \
  benchmarks/benchmark_shared_payment_cores.py \
  --batches 256 --steps-per-unit 128 --consumers 2 16 \
  --warmups 3 --repeats 15 \
  --output /tmp/jact_shared_payment_cpu_current_scaling.json
```

The horizon was four years. Each ordinary/shared pair passed the benchmark's
component-by-component comparison (`rtol=3e-5, atol=1e-6`) before timing. The
table gives median times from 15 warmed, synchronized calls. Compilation was
excluded; ordinary and shared runs alternated order.

| Batch | Duration cells | Core | Consumers | Ordinary (ms) | Shared (ms) | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 128 | cheap | 1 | 1.791 | 1.787 | 1.00x |
| 64 | 128 | derived | 1 | 2.764 | 2.640 | 1.05x |
| 64 | 128 | cheap | 8 | 2.796 | 1.445 | 1.94x |
| 64 | 128 | derived | 8 | 4.006 | 2.393 | 1.67x |
| 64 | 512 | cheap | 1 | 24.870 | 24.980 | 1.00x |
| 64 | 512 | derived | 1 | 27.530 | 25.030 | 1.10x |
| 64 | 512 | cheap | 8 | 35.744 | 15.832 | 2.26x |
| 64 | 512 | derived | 8 | 41.475 | 18.518 | 2.24x |
| 256 | 128 | cheap | 1 | 6.676 | 6.800 | 0.98x |
| 256 | 128 | derived | 1 | 9.083 | 8.521 | 1.07x |
| 256 | 128 | cheap | 8 | 9.845 | 4.682 | 2.10x |
| 256 | 128 | derived | 8 | 12.876 | 7.025 | 1.83x |
| 256 | 512 | cheap | 1 | 178.663 | 182.713 | 0.98x |
| 256 | 512 | derived | 1 | 182.868 | 184.693 | 0.99x |
| 256 | 512 | cheap | 2 | 243.538 | 186.104 | 1.31x |
| 256 | 512 | derived | 2 | 222.729 | 180.733 | 1.23x |
| 256 | 512 | cheap | 8 | 462.122 | 192.213 | 2.40x |
| 256 | 512 | derived | 8 | 471.600 | 197.259 | 2.39x |
| 256 | 512 | cheap | 16 | 724.332 | 205.408 | 3.53x |
| 256 | 512 | derived | 16 | 800.603 | 199.159 | 4.02x |

On this CPU, eight consumers gained 1.67–2.40x across the matched GPU matrix;
one consumer was close to even. At the largest grid, 16 consumers gained
3.53–4.02x, versus 1.21–1.28x on the GPU. The earlier
[CPU measurements](shared_payment_core_benchmark.md) used a different JAX
version, so their absolute timings should not be treated as a controlled
comparison with this run.
