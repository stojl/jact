# Duration-limit approximations

Implemented design; the normative contract is [api_spec.md](api_spec.md).

## Motivation

For horizon `H` and resolution `S`, the original solver takes `N = H * S`
steps and carries `N` duration cells per state and individual. Evaluating
arbitrary duration-dependent callables at every step costs roughly `O(N²)`.
Expensive fitted intensities and payments make evaluation particularly costly;
long horizons also increase probability storage and transport work.

Three independent, keyword-only solve options address these costs:

```python
result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    intensity_duration_limit=5,
    payment_duration_limit=10,
    probability_duration_limit=15,
)
```

Each defaults to `None`, preserving the original behavior when disabled. Limits
are finite nonnegative Python `int` or `float` values, excluding booleans, in
model duration units. They are static JIT configuration. `StateSpace` and model
construction remain unchanged. Both public solve entry points expose them.

## Function limits

An intensity limit `L` means `mu(t, d) ≈ mu(t, min(d, L))`. A payment limit
applies the identical rule to payment values. Clock time and cashflow weights
remain unchanged. A payment-only limit cannot change probabilities.

The solver builds separate compact midpoint and left-endpoint layouts. It
passes only retained duration samples and, when necessary, one threshold sample
to the callable, then gathers those values back to the probability cells.
Existing samples exactly at the threshold are reused; no extra grid sample is
introduced if every sample is below the limit. This is an evaluation reduction,
not merely a full-width array of clipped duration inputs.

Limits also apply to initial point-mass evaluation, same-step payments, and
duration-event payment values. Event triggering is unchanged. Available grid
threshold values are reused for point masses; batched individual durations
remain vectorized. With a mixed batch, point evaluation can still evaluate a
clipped per-individual vector, while a batch entirely past the threshold can
reuse the grid value directly.

Without probability compression, function limits preserve the complete regular
probability grid and its characteristic transport. With compression, the tail
uses the probability cutoff or the smaller applicable function cutoff.

## Probability compression

For probability cutoff `P`, retain

```text
K = min(N, floor(P * S) + 1)
```

regular cells, snapping products within `1e-10` of an integer before flooring.
Their left nodes are `k / S <= P` (up to snapping). Retention is bounded by the
original width, and includes at least the zero-duration cell.
When `K=N`, regular transport is unchanged and the tail stays empty at `P`.

The carry adds a separate tail for continuous probability:

```python
class _TailProbability(NamedTuple):
    mass: jnp.ndarray
    duration: jnp.ndarray

class StateCarry(NamedTuple):
    density: jnp.ndarray
    point_mass: _PointMass | None
    tail: _TailProbability | None = None
```

Tail mass starts at zero. Its representative duration stays at `P` forever,
even when empty. This resolves the earlier proposal's open representative-
duration question in favor of a fixed cutoff, rather than a weighted mean or
an aging cohort. Heterogeneity and further duration aging are discarded.

Regular cells retain the existing survival and characteristic shift. Survivors
that leave the last retained cell accumulate in the tail. For a one-cell grid,
the part of midpoint inflow destined for cell one enters the tail directly.
Tail survival and competing exits use the existing formulas; outgoing mass
resets duration in destination states and participates in same-step transfers.
Tail occupancy and exits contribute to state-rate, transition-lump, and
scheduled-event cashflows.

Initial point masses remain separately tracked at their true evolving
durations, including initial durations above every cutoff. Only continuous
probability is compressed.

The implementation separates `N` simulation steps from `K` retained cells in
scans, recording, event indexing, and device dispatch. Internal survival and
cashflow arithmetic temporarily appends tail mass as a final accumulating cell,
then splits it from the regular density in each returned carry. The tail's
sampling duration is fixed in both midpoint and left-endpoint layouts. JIT,
reverse-mode checkpointing, and padded batch sharding preserve this structure.

For a fixed cutoff, regular-grid arithmetic is `O(NK)` rather than `O(N²)`;
carry storage is `O(K)` rather than `O(N)` per state and individual. Function
limits alone primarily reduce callable evaluation, not probability arithmetic.

## Duration events

Duration-event snapping and horizon rules remain unchanged. Continuous event
payments come only from retained regular cells. Omitted cells contribute zero;
the tail never triggers duration events. Initial point masses still trigger
normally, including targets beyond the probability cutoff.

Scalar and callable targets, including traced targets, are accepted without
compression-specific rejection or warnings. This resolves the earlier
proposal's suggested rejection policy: missing continuous payments above the
retained grid are an explicit approximation chosen by the user. Function limits
alone preserve duration-event timing and probability, changing only payment
values where applicable.

## Outputs and diagnostics

- `StateProbability()` includes regular density, tail mass, and point masses.
- `DensityProbability()` and the density leaf of `MarginalComponents()` include
  regular density plus tail mass.
- `Density()` contains only the `K` regular cells under compression.
- `Tail()` returns `{"mass": ..., "duration": ...}` with `(T, B, S)` leaves.
  Both are zero when compression is disabled. Under compression, duration is
  always `P`, including at time zero.
- `Full()` adds the same `"tail"` payload under compression; otherwise its
  original two-key structure is unchanged.
- Custom callbacks can inspect `StateCarry.tail`. `ModelResult` does not store
  diagnostics or approximation settings automatically.

Compare each approximation against an exact solve on representative inputs.
Tail mass is a useful diagnostic but does not by itself bound errors: hazards
and payment magnitudes beyond the cutoff matter too.

## Validation and benchmarking

`tests/test_duration_limits.py` covers compact callable shapes, clamped-function
references, independent and combined controls, limit orderings, one-cell
compression, fixed-duration survival and cashflows, mass conservation, cyclic
transfers, duration events, point masses, output reducers, JIT, gradients,
recording, fitted wrappers, and padded multi-device execution. Existing tests
cover unchanged default behavior; static consumer tests check the new API.

Run the reproducible benchmark with:

```bash
JAX_PLATFORMS=cpu .venv/bin/python benchmarks/benchmark_duration_limits.py \
    --horizons 10 20 40 --batch 16 --steps-per-unit 12 --limit 2 --repeats 5 \
    --output /tmp/duration_limits.json
```

It compares exact, intensity-only, payment-only, probability-only, and combined
solves for cheap and expensive callables. Timings exclude compilation, warm each
configuration, and synchronize every timed solve. Output includes settings,
median runtime, numeric carry bytes, final mean tail mass, maximum state-
probability error, and maximum terminal-cashflow error against the exact solve.
Carry bytes exclude output buffers and temporary arrays. There are no hardware-
dependent speed assertions. See [the measured CPU report](duration_limit_benchmark.md)
for settings, timings, carry sizes, tail masses, and errors from a local run.

## Boundaries

Automatic cutoff selection, aging or weighted-mean tails, multiple buckets,
runtime event validation, and a public comparison utility remain out of scope.
