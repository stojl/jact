"""Initial state-and-duration distribution for solver entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp

ArrayLike = Any

__all__ = ["InitialDistribution"]


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return len(shape) == 0


def _as_tuple_of_unique_states(
    initial_states: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if initial_states is None:
        return None
    states = tuple(initial_states)
    if len(states) != len(set(states)):
        raise ValueError("initial_states must contain unique state names.")
    for state in states:
        if not isinstance(state, str):
            raise TypeError(
                "initial_states must contain only strings, "
                f"got {type(state)}"
            )
    return states


def _array_shape(value: ArrayLike) -> tuple[int, ...]:
    return jnp.shape(value)


def _shapes_are_broadcast_compatible(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> bool:
    return left == right or _is_scalar_shape(left) or _is_scalar_shape(right)


def _validate_non_negative_if_concrete(name: str, value: ArrayLike) -> None:
    try:
        arr = jnp.asarray(value)
        if bool(jnp.any(arr < 0)):
            raise ValueError(f"{name} must be non-negative.")
    except Exception as exc:  # pragma: no cover - tracer path
        if "tracer" not in type(exc).__name__.lower():
            try:
                message = str(exc).lower()
            except Exception:  # pragma: no cover
                message = ""
            if "tracer" not in message and "concret" not in message:
                raise


def _component_payload(
    payload: Any,
) -> tuple[ArrayLike, ArrayLike]:
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Each component payload must be a mapping containing 'mass' and "
            f"'duration', got {type(payload)}."
        )
    if "mass" not in payload or "duration" not in payload:
        raise ValueError(
            "Each component must contain both 'mass' and 'duration'."
        )
    return payload["mass"], payload["duration"]


def _validate_integer_indices_if_concrete(states: ArrayLike) -> None:
    try:
        dtype = jnp.asarray(states).dtype
    except Exception as exc:  # pragma: no cover - tracer path
        if "tracer" not in type(exc).__name__.lower():
            message = str(exc).lower()
            if "tracer" not in message and "concret" not in message:
                raise
        return

    if jnp.issubdtype(dtype, jnp.integer):
        return
    raise TypeError("per_individual states must use an integer dtype.")


def _raise_invalid_per_individual_indices(is_valid: bool) -> None:
    if not bool(is_valid):
        raise ValueError(
            "per_individual states must index into the declared "
            "initial-state set."
        )


@dataclass(frozen=True)
class _CanonicalDistribution:
    states: tuple[str, ...]
    masses: tuple[ArrayLike, ...]
    durations: tuple[ArrayLike, ...]
    batch_size: int | None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InitialDistribution:
    """Structural initial states plus runtime `(state, duration)` values.

    `InitialDistribution` has two jobs at solver entry: declare which states
    count as initial structurally, and attach runtime mass and duration values
    within that declared set. Model reduction follows the declared structure,
    not runtime mass support, so a declared zero-mass state still counts as an
    initial state structurally.
    """

    _kind: str
    _declared_states: tuple[str, ...] | None
    _normalise: bool
    _masses: tuple[ArrayLike, ...]
    _durations: tuple[ArrayLike, ...]
    _state_indices: ArrayLike | None
    _per_individual_duration: ArrayLike | None

    def __init__(
        self,
        components: Mapping[str, Mapping[str, ArrayLike]],
        normalise: bool = True,
    ):
        if not components:
            raise ValueError("components must be a non-empty mapping.")

        states: list[str] = []
        masses: list[ArrayLike] = []
        durations: list[ArrayLike] = []
        batch_size: int | None = None

        for state, payload in components.items():
            if not isinstance(state, str):
                raise TypeError(
                    "components keys must be state names (strings), "
                    f"got {type(state)}"
                )
            mass, duration = _component_payload(payload)
            mass_shape = _array_shape(mass)
            duration_shape = _array_shape(duration)
            if not _shapes_are_broadcast_compatible(mass_shape, duration_shape):
                raise ValueError(
                    f"Component '{state}' has incompatible mass and duration "
                    f"shapes: {mass_shape} vs {duration_shape}."
                )
            if not (_is_scalar_shape(mass_shape) or len(mass_shape) == 1):
                raise ValueError(
                    f"Component '{state}' must use a scalar or (batch,) array."
                )
            if not (_is_scalar_shape(duration_shape) or len(duration_shape) == 1):
                raise ValueError(
                    f"Component '{state}' must use a scalar or (batch,) array."
                )
            component_batch = None
            if not _is_scalar_shape(mass_shape):
                component_batch = mass_shape[0]
            if not _is_scalar_shape(duration_shape):
                if component_batch is None:
                    component_batch = duration_shape[0]
                elif component_batch != duration_shape[0]:
                    raise ValueError(
                        f"Component '{state}' has incompatible mass and "
                        "duration batch dimensions."
                    )
            if component_batch is not None:
                if batch_size is None:
                    batch_size = component_batch
                elif batch_size != component_batch:
                    raise ValueError(
                        "All component batch dimensions must match."
                    )

            _validate_non_negative_if_concrete("mass", mass)
            _validate_non_negative_if_concrete("duration", duration)

            states.append(state)
            masses.append(mass)
            durations.append(duration)

        object.__setattr__(self, "_kind", "components")
        object.__setattr__(self, "_declared_states", tuple(states))
        object.__setattr__(self, "_normalise", bool(normalise))
        object.__setattr__(self, "_masses", tuple(masses))
        object.__setattr__(self, "_durations", tuple(durations))
        object.__setattr__(self, "_state_indices", None)
        object.__setattr__(self, "_per_individual_duration", None)

    @classmethod
    def at(
        cls,
        state: str,
        duration: ArrayLike = 0.0,
    ) -> InitialDistribution:
        """Declare one structural initial state with all mass at `state`.

        This is the explicit single-state form of the `initial="state"`
        shorthand at solve entry. `duration` is runtime data inside that
        declared one-state structure.
        """
        return cls(
            components={state: {"mass": jnp.asarray(1.0), "duration": duration}},
            normalise=False,
        )

    @classmethod
    def per_individual(
        cls,
        states: ArrayLike,
        duration: ArrayLike = 0.0,
        initial_states: Sequence[str] | None = None,
    ) -> InitialDistribution:
        """Declare per-individual initial states by index.

        If `initial_states` is a tuple, `states` indexes into that declared
        tuple and reduction follows that structural set. If
        `initial_states is None`, `states` indexes into the model's full state
        list. In either mode, runtime index values do not change the declared
        structural initial-state set.
        """
        state_shape = _array_shape(states)
        if len(state_shape) != 1:
            raise ValueError(
                "per_individual states must be a rank-1 (batch,) array."
            )
        _validate_integer_indices_if_concrete(states)

        duration_shape = _array_shape(duration)
        if not (_is_scalar_shape(duration_shape) or len(duration_shape) == 1):
            raise ValueError(
                "per_individual duration must be scalar or a (batch,) array."
            )
        if len(duration_shape) == 1 and duration_shape[0] != state_shape[0]:
            raise ValueError(
                "per_individual duration batch dimension must match states."
            )

        _validate_non_negative_if_concrete("duration", duration)

        self = cls.__new__(cls)
        object.__setattr__(self, "_kind", "per_individual")
        object.__setattr__(
            self,
            "_declared_states",
            _as_tuple_of_unique_states(initial_states),
        )
        object.__setattr__(self, "_normalise", False)
        object.__setattr__(self, "_masses", ())
        object.__setattr__(self, "_durations", ())
        object.__setattr__(self, "_state_indices", states)
        object.__setattr__(self, "_per_individual_duration", duration)
        return self

    @property
    def declared_initial_states(self) -> tuple[str, ...] | None:
        return self._declared_states

    def tree_flatten(self):
        children = (
            *self._masses,
            *self._durations,
            self._state_indices,
            self._per_individual_duration,
        )
        aux = (
            self._kind,
            self._declared_states,
            self._normalise,
            len(self._masses),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kind, declared_states, normalise, n_components = aux
        masses = tuple(children[:n_components])
        durations = tuple(children[n_components : 2 * n_components])
        state_indices = children[2 * n_components]
        per_individual_duration = children[2 * n_components + 1]
        self = cls.__new__(cls)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_declared_states", declared_states)
        object.__setattr__(self, "_normalise", normalise)
        object.__setattr__(self, "_masses", masses)
        object.__setattr__(self, "_durations", durations)
        object.__setattr__(self, "_state_indices", state_indices)
        object.__setattr__(self, "_per_individual_duration", per_individual_duration)
        return self

    def _batch_size(self) -> int | None:
        arrays: Iterable[ArrayLike]
        if self._kind == "components":
            arrays = (*self._masses, *self._durations)
        else:
            arrays = (self._state_indices, self._per_individual_duration)

        batch_size: int | None = None
        for value in arrays:
            if value is None:
                continue
            shape = _array_shape(value)
            if len(shape) == 1:
                if batch_size is None:
                    batch_size = shape[0]
                elif batch_size != shape[0]:
                    raise ValueError(
                        "InitialDistribution batch dimensions must match."
                    )
        return batch_size

    def active_initial_states(
        self,
        model_states: Sequence[str],
    ) -> tuple[str, ...]:
        if self._kind == "per_individual" and self._declared_states is None:
            return tuple(model_states)
        if self._declared_states is None:
            raise ValueError("InitialDistribution has no declared states.")
        return self._declared_states

    def canonicalize(
        self,
        model_states: Sequence[str],
    ) -> _CanonicalDistribution:
        if self._kind == "components":
            states = self.active_initial_states(model_states)
            masses = self._broadcast_components(self._masses)
            durations = self._broadcast_components(self._durations)
            if self._normalise:
                masses = self._normalise_masses(masses)
            return _CanonicalDistribution(
                states=states,
                masses=masses,
                durations=durations,
                batch_size=self._batch_size(),
            )

        states = self.active_initial_states(model_states)
        batch_size = self._batch_size()
        if batch_size is None:
            raise ValueError(
                "per_individual requires a rank-1 states array to define batch size."
            )

        duration = self._broadcast_value(self._per_individual_duration, batch_size)
        indices = jnp.asarray(self._state_indices)
        one_hot = jax.nn.one_hot(indices, len(states), dtype=duration.dtype)
        masses = tuple(one_hot[:, i] for i in range(len(states)))
        durations = tuple(duration for _ in states)
        return _CanonicalDistribution(
            states=states,
            masses=masses,
            durations=durations,
            batch_size=batch_size,
        )

    def validate_for_model(self, model_states: Sequence[str]) -> None:
        state_set = set(model_states)
        for state in self.active_initial_states(model_states):
            if state not in state_set:
                raise ValueError(
                    f"'{state}' is not a declared state. "
                    f"Available states: {tuple(model_states)}"
                )

        if self._kind != "per_individual":
            return

        try:
            indices = jnp.asarray(self._state_indices)
            n_states = len(self.active_initial_states(model_states))
            is_valid = jnp.all((indices >= 0) & (indices < n_states))
            if not bool(is_valid):
                _raise_invalid_per_individual_indices(False)
        except Exception as exc:  # pragma: no cover - tracer path
            if "tracer" not in type(exc).__name__.lower():
                message = str(exc).lower()
                if "tracer" not in message and "concret" not in message:
                    raise
            jax.debug.callback(
                _raise_invalid_per_individual_indices,
                is_valid,
            )

    def _broadcast_components(
        self,
        values: tuple[ArrayLike, ...],
    ) -> tuple[jnp.ndarray, ...]:
        batch_size = self._batch_size()
        if batch_size is None:
            return tuple(jnp.asarray(value) for value in values)
        return tuple(self._broadcast_value(value, batch_size) for value in values)

    @staticmethod
    def _broadcast_value(value: ArrayLike, batch_size: int) -> jnp.ndarray:
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (batch_size,))
        if arr.ndim == 1:
            if arr.shape[0] != batch_size:
                raise ValueError("Batch dimensions must match.")
            return arr
        raise ValueError("Expected a scalar or (batch,) array.")

    @staticmethod
    def _normalise_masses(
        masses: tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, ...]:
        if not masses:
            return ()
        stacked = jnp.stack(tuple(jnp.asarray(mass) for mass in masses), axis=0)
        totals = jnp.sum(stacked, axis=0, keepdims=True)
        safe_totals = jnp.where(totals > 0, totals, 1.0)
        normalised = jnp.where(totals > 0, stacked / safe_totals, stacked)
        return tuple(normalised[i] for i in range(normalised.shape[0]))
