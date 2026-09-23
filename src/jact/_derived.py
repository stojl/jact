"""Static exogenous field graph and traced per-context evaluation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

Derived = Mapping[str, Callable[..., Any]]
_RESERVED = frozenset({"t", "d", "initial", "initial_duration"})


@dataclass(frozen=True, eq=False)
class _Node:
    name: str
    fn: Callable[..., Any]
    dependencies: tuple[str, ...]
    needs_time: bool
    needs_duration: bool

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                id(self.fn),
                self.dependencies,
                self.needs_time,
                self.needs_duration,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Node):
            return NotImplemented
        return (
            self.name == other.name
            and self.fn is other.fn
            and self.dependencies == other.dependencies
            and self.needs_time == other.needs_time
            and self.needs_duration == other.needs_duration
        )


@dataclass(frozen=True)
class FieldGraph:
    nodes: tuple[_Node, ...]

    @classmethod
    def build(
        cls,
        model: Derived | None,
        cashflows: Derived | None,
        solve: Derived | None,
        inputs: Mapping[str, Any],
    ) -> FieldGraph:
        definitions: dict[str, Callable[..., Any]] = {}
        for scope, mapping in (
            ("model", model),
            ("cashflow", cashflows),
            ("solve", solve),
        ):
            if mapping is None:
                continue
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{scope} derived must be a mapping.")
            for name, fn in mapping.items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"Invalid derived field name {name!r}.")
                if name in _RESERVED:
                    raise ValueError(f"Derived field {name!r} is reserved.")
                if name in definitions:
                    raise ValueError(f"Duplicate derived field name {name!r}.")
                if name in inputs:
                    raise ValueError(
                        f"Derived field {name!r} collides with a solve input."
                    )
                if not callable(fn):
                    raise TypeError(f"Derived field {name!r} must be callable.")
                definitions[name] = fn

        parameters: dict[str, tuple[str, ...]] = {}
        for name, fn in definitions.items():
            try:
                signature = inspect.signature(fn)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Derived field {name!r} needs an inspectable signature."
                ) from exc
            deps: list[str] = []
            for parameter in signature.parameters.values():
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                ):
                    raise TypeError(
                        f"Derived field {name!r} needs named parameters "
                        "without *args or **kwargs."
                    )
                if (
                    parameter.name not in definitions
                    and parameter.name not in inputs
                    and parameter.name not in {"t", "d"}
                ):
                    if parameter.default is inspect.Parameter.empty:
                        raise ValueError(
                            f"Derived field {name!r} has missing dependency "
                            f"{parameter.name!r}."
                        )
                    continue
                deps.append(parameter.name)
            parameters[name] = tuple(deps)

        ordered: list[_Node] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Cycle in derived fields involving {name!r}.")
            if name in visited:
                return
            visiting.add(name)
            for dep in parameters[name]:
                if dep in definitions:
                    visit(dep)
            visiting.remove(name)
            visited.add(name)
            deps = parameters[name]
            ancestors = {node.name: node for node in ordered}
            ordered.append(
                _Node(
                    name,
                    definitions[name],
                    deps,
                    "t" in deps
                    or any(
                        ancestors[dep].needs_time for dep in deps if dep in ancestors
                    ),
                    "d" in deps
                    or any(
                        ancestors[dep].needs_duration
                        for dep in deps
                        if dep in ancestors
                    ),
                )
            )

        for field in definitions:
            visit(field)
        return cls(tuple(ordered))

    def input_only(self, inputs: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        values = dict(inputs)
        for node in self.nodes:
            if not node.needs_time and not node.needs_duration:
                values[node.name] = jnp.asarray(
                    node.fn(**{dep: values[dep] for dep in node.dependencies})
                )
        return values


class FieldRuntime:
    """A step-local cache; graph functions execute once per requested context."""

    def __init__(self, graph: FieldGraph, inputs: dict[str, jnp.ndarray]):
        self.graph = graph
        self.inputs = inputs
        self.cache: dict[object, dict[str, jnp.ndarray]] = {}

    def resolve(
        self,
        key: object,
        t: jnp.ndarray | None = None,
        d: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        if key in self.cache:
            return self.cache[key]
        phase = key[1] if isinstance(key, tuple) and len(key) > 1 else key
        values = (
            dict(self.resolve(("time", phase), t))
            if d is not None and t is not None
            else dict(self.inputs)
        )
        if t is not None:
            values["t"] = t
        if d is not None:
            values["d"] = d
        for node in self.graph.nodes:
            if node.name in values:
                continue
            if (node.needs_time and t is None) or (node.needs_duration and d is None):
                continue
            values[node.name] = jnp.asarray(
                node.fn(**{dep: values[dep] for dep in node.dependencies})
            )
        values.pop("t", None)
        values.pop("d", None)
        self.cache[key] = values
        return values
