"""Array-compatible leaves for named solve inputs and derived fields."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def as_field_value(value: Any, name: str) -> Any:
    """Keep JAX tree structure while converting its numerical leaves to arrays.

    Homogeneous Python sequences retain their existing array-input behavior.
    """
    if isinstance(value, (jax.Array, np.ndarray, np.number)) or type(value) in (
        bool, int, float, complex, list, tuple
    ):
        try:
            return jnp.asarray(value)
        except (TypeError, ValueError):
            pass

    paths, treedef = jax.tree_util.tree_flatten_with_path(value)
    if not paths:
        return value
    leaves = []
    for path, leaf in paths:
        try:
            array = jnp.asarray(leaf)
            if array.dtype.kind in {"O", "S", "U"}:
                raise TypeError("expected a numeric array")
        except (TypeError, ValueError) as exc:
            location = jax.tree_util.keystr(path)
            raise TypeError(
                f"{name}{location} must be an array-compatible numeric leaf."
            ) from exc
        leaves.append(array)
    return treedef.unflatten(leaves)
