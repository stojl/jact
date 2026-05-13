"""Cashflow declarations and solve-time views."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import jax.numpy as jnp

Scalar = int | float
Weight = Callable[..., jnp.ndarray] | Scalar | jnp.ndarray | None

__all__ = [
    "ByKind",
    "ByState",
    "CashflowDeclaration",
    "DurationEvent",
    "Group",
    "Raw",
    "ScheduledEvent",
    "StateRate",
    "Total",
    "TransitionLump",
]


@dataclass(frozen=True)
class StateRate:
    """Payment-rate callables attached to occupied states."""

    payments: Mapping[str, Callable[..., jnp.ndarray]]


@dataclass(frozen=True)
class TransitionLump:
    """Lump-sum payment callables attached to transitions."""

    payments: Mapping[tuple[str, str], Callable[..., jnp.ndarray]]


@dataclass(frozen=True)
class ScheduledEvent:
    """State-conditioned payments at deterministic event times."""

    when: Callable[..., jnp.ndarray]
    payments: Mapping[str, Callable[..., jnp.ndarray]]


@dataclass(frozen=True)
class DurationEvent:
    """State-duration conditioned one-time payments."""

    at_durations: Mapping[str, float | Callable[..., jnp.ndarray]]
    payments: Mapping[str, Callable[..., jnp.ndarray]]


@dataclass(frozen=True)
class Raw:
    """Return one raw component or all raw components."""

    name: str | None = None
    weight: Weight = None
    terminal: bool = False


@dataclass(frozen=True)
class Group:
    """Return the sum of selected raw components."""

    members: Sequence[str]
    weight: Weight = None
    terminal: bool = False


@dataclass(frozen=True)
class Total:
    """Return the sum of all raw components."""

    weight: Weight = None
    terminal: bool = False


@dataclass(frozen=True)
class ByState:
    """Return cashflows split by reachable state."""

    weight: Weight = None
    terminal: bool = False


@dataclass(frozen=True)
class ByKind:
    """Return cashflows split by component kind."""

    weight: Weight = None
    terminal: bool = False


@dataclass(frozen=True)
class CashflowDeclaration:
    """Validated cashflow components bound to a state-space topology."""

    state_space: Any
    components: tuple[
        tuple[str, StateRate | TransitionLump | ScheduledEvent | DurationEvent],
        ...,
    ]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.components)

    def component(
        self,
        name: str,
    ) -> StateRate | TransitionLump | ScheduledEvent | DurationEvent:
        for component_name, component in self.components:
            if component_name == name:
                return component
        raise ValueError(f"Unknown cashflow component '{name}'.")


def _check_component_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("cashflow component names must be non-empty strings.")


def _check_callable(value: Any, field: str) -> None:
    if not callable(value):
        raise TypeError(f"{field} must be callable.")


def _validate_payment_mapping(payments: Any, field: str) -> Mapping[Any, Any]:
    if not isinstance(payments, Mapping) or not payments:
        raise ValueError(f"{field} must be a non-empty mapping.")
    for fn in payments.values():
        _check_callable(fn, f"{field} values")
    return dict(payments)


def _validate_at_duration_mapping(
    at_durations: Any,
    field: str,
) -> Mapping[Any, Any]:
    if not isinstance(at_durations, Mapping) or not at_durations:
        raise ValueError(f"{field} must be a non-empty mapping.")
    for at_duration in at_durations.values():
        if not (callable(at_duration) or _is_scalar_array_like(at_duration)):
            raise TypeError(f"{field} values must be scalar or callable.")
    return dict(at_durations)


def _validate_state_payments(state_space: Any, payments: Mapping[Any, Any]) -> None:
    for state in payments:
        state_space._check_state(state)


def _is_scalar_array_like(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(jnp.asarray(value).ndim == 0)
    except Exception:
        return False


def _normalise_weight(weight: Any) -> Any:
    if weight is None:
        return None
    if _is_scalar_array_like(weight):
        return jnp.asarray(weight).item()
    return weight


def validate_cashflow_components(
    state_space: Any,
    components: Mapping[
        str,
        StateRate | TransitionLump | ScheduledEvent | DurationEvent,
    ],
) -> CashflowDeclaration:
    """Validate and freeze a component mapping for a state space."""
    if not isinstance(components, Mapping) or not components:
        raise ValueError("cashflows() requires a non-empty component mapping.")

    frozen: list[
        tuple[str, StateRate | TransitionLump | ScheduledEvent | DurationEvent]
    ] = []
    seen: set[str] = set()
    for name, component in components.items():
        _check_component_name(name)
        if name in seen:
            raise ValueError(f"Duplicate cashflow component name '{name}'.")
        seen.add(name)

        if isinstance(component, StateRate):
            payments = _validate_payment_mapping(
                component.payments,
                f"StateRate('{name}').payments",
            )
            _validate_state_payments(state_space, payments)
            frozen_component = StateRate(payments=payments)
        elif isinstance(component, TransitionLump):
            payments = _validate_payment_mapping(
                component.payments,
                f"TransitionLump('{name}').payments",
            )
            for transition in payments:
                if (
                    not isinstance(transition, tuple)
                    or len(transition) != 2
                    or not state_space.has_transition(*transition)
                ):
                    raise ValueError(
                        f"TransitionLump('{name}') references unknown "
                        f"transition {transition!r}."
                    )
            frozen_component = TransitionLump(payments=payments)
        elif isinstance(component, ScheduledEvent):
            _check_callable(component.when, f"ScheduledEvent('{name}').when")
            payments = _validate_payment_mapping(
                component.payments,
                f"ScheduledEvent('{name}').payments",
            )
            _validate_state_payments(state_space, payments)
            frozen_component = ScheduledEvent(
                when=component.when,
                payments=payments,
            )
        elif isinstance(component, DurationEvent):
            at_durations = _validate_at_duration_mapping(
                component.at_durations,
                f"DurationEvent('{name}').at_durations",
            )
            payments = _validate_payment_mapping(
                component.payments,
                f"DurationEvent('{name}').payments",
            )
            for state in at_durations:
                state_space._check_state(state)
            _validate_state_payments(state_space, payments)
            if set(at_durations) != set(payments):
                raise ValueError(
                    f"DurationEvent('{name}').at_durations and payments "
                    "must use the same state keys."
                )
            frozen_component = DurationEvent(
                at_durations=at_durations,
                payments=payments,
            )
        else:
            raise TypeError(
                "cashflow components must be StateRate, TransitionLump, "
                f"ScheduledEvent, or DurationEvent; got {type(component)}."
            )
        frozen.append((name, frozen_component))

    return CashflowDeclaration(state_space=state_space, components=tuple(frozen))


def _validate_view_common(view: Any) -> None:
    if not isinstance(view.terminal, bool):
        raise TypeError("cashflow view terminal must be a bool.")
    weight = view.weight
    if weight is None or callable(weight) or isinstance(weight, Number):
        return
    if _is_scalar_array_like(weight):
        return
    raise TypeError("cashflow view weight must be None, a scalar, or callable.")


def _normalised_view_kwargs(view: Any) -> dict[str, Any]:
    _validate_view_common(view)
    return {
        "weight": _normalise_weight(view.weight),
        "terminal": view.terminal,
    }


def validate_cashflow_views(
    declaration: CashflowDeclaration,
    views: Mapping[str, Raw | Group | Total | ByState | ByKind] | None,
) -> tuple[tuple[str, Raw | Group | Total | ByState | ByKind], ...]:
    """Validate and freeze solve-time cashflow views."""
    if views is None:
        views = {"raw": Raw()}
    if not isinstance(views, Mapping):
        raise TypeError("cashflow_views must be a mapping or None.")

    component_names = set(declaration.names)
    frozen: list[tuple[str, Raw | Group | Total | ByState | ByKind]] = []
    seen: set[str] = set()
    for name, view in views.items():
        if not isinstance(name, str) or not name:
            raise ValueError("cashflow view names must be non-empty strings.")
        if name in seen:
            raise ValueError(f"Duplicate cashflow view name '{name}'.")
        seen.add(name)

        if isinstance(view, Raw):
            view = Raw(
                name=view.name,
                **_normalised_view_kwargs(view),
            )
            if view.name is not None and view.name not in component_names:
                raise ValueError(
                    f"Raw view '{name}' references unknown component "
                    f"'{view.name}'."
                )
        elif isinstance(view, Group):
            if isinstance(view.members, str) or not view.members:
                raise ValueError(f"Group view '{name}' requires members.")
            members = tuple(view.members)
            for member in members:
                if member not in component_names:
                    raise ValueError(
                        f"Group view '{name}' references unknown component "
                        f"'{member}'."
                    )
            view = Group(
                members=members,
                **_normalised_view_kwargs(view),
            )
        elif isinstance(view, (Total, ByState, ByKind)):
            view = type(view)(**_normalised_view_kwargs(view))
        else:
            raise TypeError(
                "cashflow views must be Raw, Group, Total, ByState, or ByKind; "
                f"got {type(view)}."
            )
        frozen.append((name, view))
    return tuple(frozen)
