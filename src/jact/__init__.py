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
>>> result = model.solve(
...     initial="healthy",
...     horizon=10,
...     steps_per_unit=12,
...     probability=jact.probability.StateProbability(),
...     age=ages,
... )
"""

from importlib.metadata import PackageNotFoundError, version

from . import cashflows, probability, wrappers
from .initial_distribution import InitialDistribution
from .model import Model
from .result import ModelResult
from .solver import solve
from .state_space import StateSpace

try:
    __version__ = version("jact")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "StateSpace",
    "Model",
    "InitialDistribution",
    "ModelResult",
    "solve",
    "cashflows",
    "probability",
    "wrappers",
]
