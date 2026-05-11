# jact

JAX-based transition probability and expected cashflow computation for multi-state models with duration-dependent transition intensities.

## What is jact?

`jact` computes transition probabilities and expected cashflows in semi-Markov multi-state models. It takes fitted intensity models — parametric functions, GLMs, neural networks, or any JIT-compatible callable — and produces transition probabilities and cashflow streams for thousands of individuals in a single vectorized pass on GPU. Computations are optimized for JIT-compiled GPU execution.

## Quick example

```python
import jax.numpy as jnp
import jact

# Define the state space
state_space = jact.StateSpace(
    states=["healthy", "disabled", "dead"],
    transitions=[
        ("healthy", "disabled"),
        ("healthy", "dead"),
        ("disabled", "dead"),
    ],
)

# Build a model with intensity functions
model = state_space.build(
    transitions={
        ("healthy", "disabled"): onset_fn,
        ("healthy", "dead"): mortality_fn,
        ("disabled", "dead"): disabled_mort_fn,
    }
)

# Compute transition probabilities for 1000 individuals
ages = jnp.linspace(30, 80, 1_000)
result = model.solve(initial="healthy", horizon=30, steps_per_unit=12, age=ages)
```

## Fitted-model intensity wrappers

Use `bind_intensity()` when a fitted model has a separate feature builder and
apply function:

```python
def features(t, d, *, age):
    return jnp.stack(
        [
            jnp.broadcast_to(age[:, None] + t, (age.shape[0], d.shape[-1])),
            jnp.broadcast_to(d, (age.shape[0], d.shape[-1])),
        ],
        axis=-1,
    )


def apply(params, x):
    linear = params["intercept"] + jnp.sum(x * params["coef"], axis=-1)
    return jnp.exp(linear)


onset_fn = jact.bind_intensity(apply, fitted_params, features)
```

For fitted models that emit several hazards at once, use
`bind_grouped_intensity(..., output_count=K)` or
`bind_exit_intensity(..., output_count=K)`. The wrappers clamp outputs to
non-negative hazards and normalize grouped outputs to `(K, batch, D)`.

## Cashflow example

Using the same `state_space`, `model`, and `ages` as above:

```python
import jax.numpy as jnp
import jact


def annual_premium(t, d, *, age):
    return jnp.full((age.shape[0], d.shape[-1]), -1_200.0)


def death_benefit(t, d, *, age):
    return jnp.full((age.shape[0], d.shape[-1]), 100_000.0)


cashflows = state_space.cashflows(
    {
        "premium": jact.cashflows.StateRate({"healthy": annual_premium}),
        "death_benefit": jact.cashflows.TransitionLump(
            {
                ("healthy", "dead"): death_benefit,
                ("disabled", "dead"): death_benefit,
            }
        ),
    }
)

result = model.solve(
    initial="healthy",
    horizon=30,
    steps_per_unit=12,
    record_every=12,
    probability=None,
    cashflows=cashflows,
    cashflow_views={
        "raw": jact.cashflows.Raw(),
        "pv_total": jact.cashflows.Total(
            weight=lambda t, **kwargs: jnp.exp(-0.03 * t),
            terminal=True,
        ),
    },
    age=ages,
)

premium_stream = result.cashflows["raw"]["premium"]
present_value = result.cashflows["pv_total"]
```

## Key features

- **Plug in any model**: Gompertz, GLM, neural network — anything that's JIT-compatible.
- **Swap and compare**: Same `StateSpace`, different intensity models. Experiment easily.
- **Probabilities and cashflows together**: Emit both in one fused solve, with solve-time cashflow views for grouping and valuation.
- **Compute only what's needed**: The solver reduces to states reachable from the initial state.
- **Exact seeded starts**: Initial point masses preserve per-individual starting duration `d_0` exactly.
- **Batch-first**: Designed for 100K+ individuals in a single pass.

## Documentation

See the [documentation index](https://github.com/stojl/jact/blob/main/docs/index.md)
for the public documentation set.
For the full API contract, use the
[API specification](https://github.com/stojl/jact/blob/main/docs/api_spec.md).
For a runnable walkthrough of the main workflow, see the
[example notebook](https://github.com/stojl/jact/blob/main/docs/example_notebook.ipynb).
For a fitting-to-solver workflow with neural-network intensities, see the
[fitted neural-network notebook](https://github.com/stojl/jact/blob/main/docs/fitted_nn_notebook.ipynb).

## Namespace

The top-level `jact` namespace exposes the core types: `jact.StateSpace`,
`jact.Model`, `jact.InitialDistribution`, `jact.ModelResult`, and
`jact.solve`. It also exposes fitted-model helpers:
`jact.bind_intensity`, `jact.bind_grouped_intensity`, and
`jact.bind_exit_intensity`. Domain types live under two submodules:

- `jact.cashflows` for declarations and views (`StateRate`,
  `TransitionLump`, `ScheduledEvent`, `DurationEvent`, `Raw`, `Group`,
  `Total`, `ByState`, `ByKind`).
- `jact.probability` for output reducers (`StateProbability`,
  `DensityProbability`, `Density`, `PointMass`, `MarginalComponents`,
  `Full`).

Advanced inspection types stay in private modules — for example
`jact.probability.StateCarry` and `jact.model.ReducedModel`.

## Installation

```bash
pip install jax jaxlib
pip install jact
```

For local development from this repository:

```bash
pip install -e '.[dev]'
pytest
```

The package uses a `src/` layout, so editable install is the intended local
workflow.

To run the example notebook with plotting support from a local checkout:

```bash
pip install -e '.[dev,notebook]'
```

## Release checks

Before cutting a PyPI release:

```bash
rm -rf build dist src/*.egg-info
python -m build --no-isolation
python -m twine check dist/*
pytest -q
```

The tag-driven publish flow is documented in [RELEASING.md](RELEASING.md).

## Requirements

- Python >= 3.10
- JAX >= 0.4

## License

Apache-2.0
