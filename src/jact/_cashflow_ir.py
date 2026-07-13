# pyright: strict, reportMissingImports=false
"""Private typed intermediate representation for solver cashflows."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

import jax.numpy as jnp

from .cashflows import Scalar
from .typing import DurationAt, Payment, Weight, When


class StatePayment(NamedTuple):
    state_index: int
    payment: Payment


class TransitionPayment(NamedTuple):
    source_index: int
    hazard_slot: int
    payment: Payment


DurationTarget: TypeAlias = DurationAt | int | float | complex | bool


class DurationTargetPayment(NamedTuple):
    state_index: int
    at_duration: DurationTarget
    payment: Payment


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
FormattedCashflowValue: TypeAlias = jnp.ndarray | dict[str, jnp.ndarray]
FormattedCashflows: TypeAlias = dict[str, FormattedCashflowValue]


class StepAggregation(NamedTuple):
    by_component: CashflowLeaves
    by_state: CashflowLeaves
    by_kind: CashflowLeaves
    event_by_component: CashflowLeaves
    event_by_state: CashflowLeaves
    event_by_kind: CashflowLeaves
