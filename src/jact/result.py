# pyright: strict, reportMissingImports=false, reportUntypedClassDecorator=false

"""Typed result of `Model.solve()`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar, cast

import jax

ProbabilityT = TypeVar("ProbabilityT")

CashflowGroup: TypeAlias = dict[str, jax.Array]
CashflowValue: TypeAlias = jax.Array | CashflowGroup
CashflowResult: TypeAlias = dict[str, CashflowValue]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModelResult(Generic[ProbabilityT]):
    """Typed result of `Model.solve()`.

    Attributes
    ----------
    states : tuple[str, ...]
        Reachable states in reduced order. Always present.
    probability : ProbabilityT
        Probability output, or ``None`` when ``probability=None`` was
        passed to ``solve()``. Shape depends on the chosen callback;
        time is the leading axis of every leaf.
    cashflows : dict[str, ndarray or dict[str, ndarray]] or None
        Mapping from cashflow view name to view value, or ``None`` when
        ``cashflows=None`` was passed to ``solve()``. View values are
        arrays or nested dicts; see ``docs/api_spec.md`` for the per-view
        shape table.
    """

    states: tuple[str, ...]
    probability: ProbabilityT = cast(ProbabilityT, None)
    cashflows: CashflowResult | None = None

    def tree_flatten(
        self,
    ) -> tuple[tuple[Any, Any], tuple[str, ...]]:
        return (self.probability, self.cashflows), self.states

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[str, ...],
        children: tuple[Any, Any],
    ) -> ModelResult[ProbabilityT]:
        probability, cashflows = children
        return cls(states=aux, probability=probability, cashflows=cashflows)

    def __repr__(self) -> str:
        def summarize(x: Any) -> str:
            if x is None:
                return "None"
            if isinstance(x, dict):
                values = cast(dict[object, Any], x)
                inner = ", ".join(
                    f"{k!r}: {summarize(v)}" for k, v in values.items()
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
