# pyright: strict, reportMissingImports=false
"""Private typed intermediate representation for solver cashflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeAlias

import jax.numpy as jnp

from .cashflows import Scalar
from .result import CashflowResult, CashflowValue
from .typing import ArrayLike, DurationAt, Payment, Weight, When


@dataclass(frozen=True, eq=False)
class IdentityCallable:
    """Keep callable definitions in static JIT keys without value equality."""

    fn: Callable[..., ArrayLike]

    def __hash__(self) -> int:
        return id(self.fn)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityCallable) and self.fn is other.fn

    def __call__(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self.fn(*args, **kwargs)


class StatePayment(NamedTuple):
    state_index: int
    payment: Payment
    task_id: int = -1
    weight: Weight | Scalar | None = None
    label: str = "Scaled payment"


class TransitionPayment(NamedTuple):
    source_index: int
    hazard_slot: int
    payment: Payment
    task_id: int = -1
    weight: Weight | Scalar | None = None
    label: str = "Scaled payment"


DurationTarget: TypeAlias = DurationAt | int | float | complex | bool


class DurationTargetPayment(NamedTuple):
    state_index: int
    at_duration: DurationTarget
    payment: Payment
    task_id: int = -1
    weight: Weight | Scalar | None = None
    label: str = "Scaled payment"


class StateRateSpec(NamedTuple):
    payments: tuple[StatePayment, ...]


class TransitionLumpSpec(NamedTuple):
    payments: tuple[TransitionPayment, ...]


class ScheduledEventSpec(NamedTuple):
    when: When
    payments: tuple[StatePayment, ...]


class DurationEventSpec(NamedTuple):
    targets: tuple[DurationTargetPayment, ...]


CashflowComponentSpec: TypeAlias = (
    StateRateSpec | TransitionLumpSpec | ScheduledEventSpec | DurationEventSpec
)
CashflowComponentSpecs: TypeAlias = tuple[CashflowComponentSpec, ...]


class ResolvedScheduledEvent(NamedTuple):
    event_time: jnp.ndarray
    event_index: jnp.ndarray


ResolvedScheduledEvents: TypeAlias = tuple[ResolvedScheduledEvent | None, ...]


class ResolvedDurationTarget(NamedTuple):
    state_index: int
    at_duration: jnp.ndarray
    at_duration_index: jnp.ndarray
    effective_at_duration: jnp.ndarray
    payment: Payment
    task_id: int = -1
    weight: Weight | Scalar | None = None
    label: str = "Scaled payment"


class ResolvedDurationEvent(NamedTuple):
    targets: tuple[ResolvedDurationTarget, ...]


ResolvedDurationEvents: TypeAlias = tuple[ResolvedDurationEvent | None, ...]


class ComponentSource(NamedTuple):
    component_index: int


class ComponentSumSource(NamedTuple):
    component_indices: tuple[int, ...]


class StateSource(NamedTuple):
    state_index: int


class KindSource(NamedTuple):
    kind_index: int


class TotalSource(NamedTuple):
    pass


CashflowViewSource: TypeAlias = (
    ComponentSource | ComponentSumSource | StateSource | KindSource | TotalSource
)
CashflowViewOutput: TypeAlias = Literal["single", "mapping"]


class PreparedCashflowView(NamedTuple):
    name: str
    terminal: bool
    weight: Weight | Scalar | None
    sources: tuple[CashflowViewSource, ...]
    leaf_names: tuple[str, ...]
    output: CashflowViewOutput


PreparedCashflowViews: TypeAlias = tuple[PreparedCashflowView, ...]
CashflowLeaves: TypeAlias = tuple[jnp.ndarray, ...]
CashflowViewValues: TypeAlias = tuple[CashflowLeaves, ...]
CashflowStreamValues: TypeAlias = tuple[CashflowLeaves | None, ...]
FormattedCashflowValue: TypeAlias = CashflowValue
FormattedCashflows: TypeAlias = CashflowResult


class StepAggregation(NamedTuple):
    by_component: CashflowLeaves
    by_state: CashflowLeaves
    by_kind: CashflowLeaves
    event_by_component: CashflowLeaves
    event_by_state: CashflowLeaves
    event_by_kind: CashflowLeaves
