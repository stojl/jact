"""Experimental pure-JAX fitting companion for JACT intensity models.

The fitting surface is deliberately isolated from the solver runtime. It uses
only JAX and NumPy (which JAX depends on); optimizer checkpoints and data
loaders are not part of exported :class:`FittedArtifact` objects.
"""

from .adapter import JACTAdapter
from .artifact import FittedArtifact
from .data import CanonicalBatch
from .features import CategoricalFeature, FeatureEncoder, NumericFeature
from .likelihood import (
    LikelihoodResult,
    fixed_quadrature_nll,
    piecewise_exponential_nll,
)
from .structured import (
    StructuredLogHazardModel,
    StructuredModelConfig,
    empirical_intercepts,
    linear_spline_basis,
)
from .topology import TopologySpec
from .training import (
    AdamWConfig,
    FitResult,
    Trainer,
    TrainingHistory,
    TrainState,
    adamw_init,
    make_train_step,
)

__all__ = [
    "AdamWConfig",
    "CanonicalBatch",
    "CategoricalFeature",
    "FeatureEncoder",
    "FitResult",
    "FittedArtifact",
    "JACTAdapter",
    "LikelihoodResult",
    "NumericFeature",
    "StructuredLogHazardModel",
    "StructuredModelConfig",
    "TopologySpec",
    "Trainer",
    "TrainingHistory",
    "TrainState",
    "adamw_init",
    "empirical_intercepts",
    "fixed_quadrature_nll",
    "linear_spline_basis",
    "make_train_step",
    "piecewise_exponential_nll",
]
