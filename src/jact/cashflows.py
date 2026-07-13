# pyright: strict, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportPrivateUsage=false
"""Cashflow declarations and solve-time views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, TypeAlias, cast

import jax.numpy as jnp
from jax.typing import ArrayLike

from .state_space import StateSpace
from .typing import DurationAt, Payment, Weight, When

Scalar = bool | int | float | complex

__all__ = [
    "ByKind",
    "ByState",
    "CashflowComponent",
    "CashflowDeclaration",
    "CashflowView",
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

    payments: Mapping[str, Payment]


@dataclass(frozen=True)
class TransitionLump:
    """Lump-sum payment callables attached to transitions."""

    payments: Mapping[tuple[str, str], Payment]


@dataclass(frozen=True)
class ScheduledEvent:
    """State-conditioned payments at deterministic event times."""

    when: When
    payments: Mapping[str, Payment]


@dataclass(frozen=True)
class DurationEvent:
    """State-duration conditioned one-time payments."""

    at_durations: Mapping[str, ArrayLike | DurationAt]
    payments: Mapping[str, Payment]


@dataclass(frozen=True)
class Raw:
    """Return one raw component or all raw components."""

    name: str | None = None
    weight: Weight | Scalar | ArrayLike | None = None
    terminal: bool = False


@dataclass(frozen=True)
class Group:
    """Return the sum of selected raw components."""

    members: Sequence[str]
    weight: Weight | Scalar | ArrayLike | None = None
    terminal: bool = False


@dataclass(frozen=True)
class Total:
    """Return the sum of all raw components."""

    weight: Weight | Scalar | ArrayLike | None = None
    terminal: bool = False


@dataclass(frozen=True)
class ByState:
    """Return cashflows split by reachable state."""

    weight: Weight | Scalar | ArrayLike | None = None
    terminal: bool = False


@dataclass(frozen=True)
class ByKind:
    """Return cashflows split by component kind."""

    weight: Weight | Scalar | ArrayLike | None = None
    terminal: bool = False


CashflowComponent: TypeAlias = (
    StateRate | TransitionLump | ScheduledEvent | DurationEvent
)
CashflowView: TypeAlias = Raw | Group | Total | ByState | ByKind


@dataclass(frozen=True)
class CashflowDeclaration:
    """Validated cashflow components bound to a state-space topology."""

    state_space: StateSpace
    components: tuple[tuple[str, CashflowComponent], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.components)

    def component(
        self,
        name: str,
    ) -> CashflowComponent:
        for component_name, component in self.components:
            if component_name == name:
                return component
        raise ValueError(f"Unknown cashflow component '{name}'.")


def _check_component_name(name: object) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("cashflow component names must be non-empty strings.")


def _check_callable(value: object, field: str) -> None:
    if not callable(value):
        raise TypeError(f"{field} must be callable.")


def _validate_payment_mapping(
    payments: object,
    field: str,
) -> dict[object, Payment]:
    if not isinstance(payments, Mapping) or not payments:
        raise ValueError(f"{field} must be a non-empty mapping.")
    for fn in payments.values():
        _check_callable(fn, f"{field} values")
    return {
        key: cast(Payment, fn)
        for key, fn in cast(Mapping[object, object], payments).items()
    }


def _validate_at_duration_mapping(
    at_durations: object,
    field: str,
) -> dict[object, ArrayLike | DurationAt]:
    if not isinstance(at_durations, Mapping) or not at_durations:
        raise ValueError(f"{field} must be a non-empty mapping.")
    normalised: dict[object, ArrayLike | DurationAt] = {}
    for state, at_duration in cast(
        Mapping[object, object], at_durations
    ).items():
        if callable(at_duration):
            normalised[state] = cast(DurationAt, at_duration)
        elif _is_scalar_array_like(at_duration):
            normalised[state] = jnp.asarray(at_duration).item()
        else:
            raise TypeError(f"{field} values must be scalar or callable.")
    return normalised


def _validate_state_payments(
    state_space: StateSpace,
    payments: Mapping[object, Payment],
) -> None:
    for state in payments:
        state_space._check_state(cast(str, state))


def _is_scalar_array_like(value: object) -> bool:
    if value is None:
        return False
    try:
        return bool(jnp.asarray(value).ndim == 0)
    except Exception:
        return False


def _normalise_weight(
    weight: Weight | Scalar | ArrayLike | None,
) -> Weight | Scalar | None:
    if weight is None:
        return None
    if _is_scalar_array_like(weight):
        return cast(Scalar, jnp.asarray(weight).item())
    return cast(Weight, weight)


def validate_cashflow_components(
    state_space: StateSpace,
    components: Mapping[str, CashflowComponent],
) -> CashflowDeclaration:
    """Validate and freeze a component mapping for a state space."""
    if not isinstance(components, Mapping) or not components:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("cashflows() requires a non-empty component mapping.")

    frozen: list[tuple[str, CashflowComponent]] = []
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
            frozen_component = StateRate(
                payments=cast(Mapping[str, Payment], payments)
            )
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
            frozen_component = TransitionLump(
                payments=cast(Mapping[tuple[str, str], Payment], payments)
            )
        elif isinstance(component, ScheduledEvent):
            _check_callable(component.when, f"ScheduledEvent('{name}').when")
            payments = _validate_payment_mapping(
                component.payments,
                f"ScheduledEvent('{name}').payments",
            )
            _validate_state_payments(state_space, payments)
            frozen_component = ScheduledEvent(
                when=component.when,
                payments=cast(Mapping[str, Payment], payments),
            )
        elif isinstance(component, DurationEvent):  # pyright: ignore[reportUnnecessaryIsInstance]
            at_durations = _validate_at_duration_mapping(
                component.at_durations,
                f"DurationEvent('{name}').at_durations",
            )
            payments = _validate_payment_mapping(
                component.payments,
                f"DurationEvent('{name}').payments",
            )
            for state in at_durations:
                state_space._check_state(cast(str, state))
            _validate_state_payments(state_space, payments)
            if set(at_durations) != set(payments):
                raise ValueError(
                    f"DurationEvent('{name}').at_durations and payments "
                    "must use the same state keys."
                )
            frozen_component = DurationEvent(
                at_durations=cast(
                    Mapping[str, ArrayLike | DurationAt], at_durations
                ),
                payments=cast(Mapping[str, Payment], payments),
            )
        else:
            raise TypeError(
                "cashflow components must be StateRate, TransitionLump, "
                f"ScheduledEvent, or DurationEvent; got {type(component)}."
            )
        frozen.append((name, frozen_component))

    return CashflowDeclaration(state_space=state_space, components=tuple(frozen))


def _validate_view_common(view: CashflowView) -> None:
    if not isinstance(view.terminal, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("cashflow view terminal must be a bool.")
    weight = view.weight
    if weight is None or callable(weight) or isinstance(weight, Number):
        return
    if _is_scalar_array_like(weight):
        return
    raise TypeError("cashflow view weight must be None, a scalar, or callable.")


class _NormalisedView(NamedTuple):
    weight: Weight | Scalar | None
    terminal: bool


def _normalised_view(view: CashflowView) -> _NormalisedView:
    _validate_view_common(view)
    return _NormalisedView(
        weight=_normalise_weight(view.weight),
        terminal=view.terminal,
    )


def validate_cashflow_views(
    declaration: CashflowDeclaration,
    views: Mapping[str, CashflowView] | None,
) -> tuple[tuple[str, CashflowView], ...]:
    """Validate and freeze solve-time cashflow views."""
    if views is None:
        views = {"raw": Raw()}
    if not isinstance(views, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("cashflow_views must be a mapping or None.")

    component_names = set(declaration.names)
    frozen: list[tuple[str, CashflowView]] = []
    seen: set[str] = set()
    for name, view in views.items():
        if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("cashflow view names must be non-empty strings.")
        if name in seen:
            raise ValueError(f"Duplicate cashflow view name '{name}'.")
        seen.add(name)

        if isinstance(view, Raw):
            common = _normalised_view(view)
            view = Raw(
                name=view.name,
                weight=common.weight,
                terminal=common.terminal,
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
            common = _normalised_view(view)
            view = Group(
                members=members,
                weight=common.weight,
                terminal=common.terminal,
            )
        elif isinstance(view, (Total, ByState, ByKind)):  # pyright: ignore[reportUnnecessaryIsInstance]
            common = _normalised_view(view)
            if isinstance(view, Total):
                view = Total(weight=common.weight, terminal=common.terminal)
            elif isinstance(view, ByState):
                view = ByState(weight=common.weight, terminal=common.terminal)
            else:
                view = ByKind(weight=common.weight, terminal=common.terminal)
        else:
            raise TypeError(
                "cashflow views must be Raw, Group, Total, ByState, or ByKind; "
                f"got {type(view)}."
            )
        frozen.append((name, view))
    return tuple(frozen)
