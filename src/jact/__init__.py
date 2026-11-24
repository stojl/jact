# pyright: strict, reportMissingImports=false

"""jact — JAX-based transition probability computation for multi-state models.

A framework for computing transition probabilities in semi-Markov
multi-state models with duration-dependent transition intensities.

Example
-------
>>> import jact
>>>
>>> model = jact.Model(
...     states=["healthy", "dead"],
...     transitions={("healthy", "dead"): lambda t, d, **kw: 0.01},
... )
>>>
>>> result = model.solve(initial="healthy", horizon=1, steps_per_unit=12)
>>> result.states
('healthy', 'dead')
"""

from importlib.metadata import PackageNotFoundError, version

from . import cashflows, probability, typing, wrappers
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
    "typing",
    "wrappers",
]
