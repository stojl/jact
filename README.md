# jact

JAX-based transition probability computation for multi-state models with duration-dependent transition intensities.

## What is jact?

`jact` computes transition probabilities in semi-Markov multi-state models. It takes fitted intensity models — parametric functions, GLMs, neural networks, or any JIT-compatible callable — and produces transition probabilities for 100K+ individuals in a single vectorized pass on GPU.

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

# Compute transition probabilities for 100K individuals
ages = jnp.linspace(30, 80, 100_000)
result = model.solve(initial="healthy", horizon=30, steps_per_unit=12, age=ages)
```

## Key features

- **Plug in any model**: Gompertz, GLM, neural network — anything that's JIT-compatible.
- **Swap and compare**: Same `StateSpace`, different intensity models. Experiment easily.
- **Compute only what's needed**: The solver reduces to states reachable from the initial state.
- **Batch-first**: Designed for 100K+ individuals in a single pass.

## Documentation

See [docs/api_spec.md](docs/api_spec.md) for the full API specification.

## Installation

```bash
pip install jax jaxlib
pip install -e .
```

## Requirements

- Python >= 3.10
- JAX >= 0.4
