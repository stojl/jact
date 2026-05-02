"""Typed result of `Model.solve()`."""

from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelResult:
    """Typed result of `Model.solve()`.

    Attributes
    ----------
    states : tuple[str, ...]
        Reachable states in reduced order. Always present.
    probability : Any or None
        Probability output, or ``None`` when ``probability=None`` was
        passed to ``solve()``. Shape depends on the chosen callback;
        time is the leading axis of every leaf.
    cashflows : dict[str, Any] or None
        Mapping from cashflow view name to view value, or ``None`` when
        ``cashflows=None`` was passed to ``solve()``. View values are
        arrays or nested dicts; see ``docs/api_spec.md`` for the per-view
        shape table.
    """

    states: tuple[str, ...]
    probability: Any = None
    cashflows: Any = None

    def tree_flatten(self):
        return (self.probability, self.cashflows), self.states

    @classmethod
    def tree_unflatten(cls, aux, children):
        probability, cashflows = children
        return cls(states=aux, probability=probability, cashflows=cashflows)

    def __repr__(self) -> str:
        def summarize(x: Any) -> str:
            if x is None:
                return "None"
            if isinstance(x, dict):
                inner = ", ".join(
                    f"{k!r}: {summarize(v)}" for k, v in x.items()
                )
                return "{" + inner + "}"
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return f"Array(shape={tuple(x.shape)}, dtype={x.dtype})"
            return repr(x)

        return (
            f"ModelResult(states={self.states!r}, "
            f"probability={summarize(self.probability)}, "
            f"cashflows={summarize(self.cashflows)})"
        )
