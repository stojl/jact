"""jact — JAX-based transition probability computation for multi-state models.

A framework for computing transition probabilities in semi-Markov
multi-state models with duration-dependent transition intensities.

Example
-------
>>> import jact
>>> import jax.numpy as jnp
>>>
>>> state_space = jact.StateSpace(
...     states=["healthy", "disabled", "dead"],
...     transitions=[
...         ("healthy", "disabled"),
...         ("healthy", "dead"),
...         ("disabled", "dead"),
...     ],
... )
>>>
>>> model = state_space.build(
...     transitions={
...         ("healthy", "disabled"): onset_fn,
...         ("healthy", "dead"): mortality_fn,
...         ("disabled", "dead"): disabled_mort_fn,
...     }
... )
>>>
>>> result = model.solve(horizon=10, steps_per_unit=12, age=ages)
"""

from . import callbacks
from .model import Model
from .solver import solve
from .state_space import StateSpace

__all__ = [
    "StateSpace",
    "Model",
    "solve",
    "callbacks",
]
