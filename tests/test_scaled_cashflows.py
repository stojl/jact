"""Named payment cores share integration while preserving payment semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jact
from jact import cashflows as cf
from jact._cashflow_ir import DurationEventSpec
from jact.solver import _prepare_cashflow_components


def _cashflows(result: jact.ModelResult[Any]) -> dict[str, Any]:
    assert result.cashflows is not None
    return cast(dict[str, Any], result.cashflows)


def _core(t, d, **kw):
    return (1 + t) * jnp.exp(-d) * kw["scale"][:, None]


def _weight(t, **kw):
    return kw["indexed_salary"]


def _when(**kw):
    return kw["event_time"]


def _at(**kw):
    return kw["target_duration"]


def _hazard(t, d, **kw):
    return kw["hazard"][:, None] * (1 + 0.1 * d)


def _component(kind, payment):
    if kind == "rate":
        return cf.StateRate({"a": payment, "b": payment})
    if kind == "lump":
        return cf.TransitionLump({("a", "b"): payment, ("b", "c"): payment})
    if kind == "scheduled":
        return cf.ScheduledEvent(_when, {"a": payment, "b": payment})
    return cf.DurationEvent({"a": _at, "b": _at}, {"a": payment, "b": payment})


def _declaration(space, scaled):
    components: dict[str, cf.CashflowComponent] = {}
    for kind in ("rate", "lump", "scheduled", "duration"):
        # Independent construction sites refer to the same authoritative core.
        for name, weight in (("income", _weight), ("expense", -0.25), ("zero", 0.0)):
            payment: jact.typing.Payment | cf.Scaled
            if scaled:
                payment = cf.Scaled("benefit", weight=weight)
            else:

                def ordinary(t, d, *, _w=weight, **kw):
                    factor = _w(t, **kw) if callable(_w) else _w
                    return _core(t, d, **kw) * jnp.asarray(factor).reshape(-1, 1)

                payment = ordinary
            components[f"{kind}_{name}"] = _component(kind, payment)
    components["ordinary"] = cf.StateRate({"a": _core})
    components["unscaled"] = cf.StateRate(
        {
            "b": cf.Scaled("benefit") if scaled else _core,
        }
    )
    return space.cashflows(
        components,
        cores={"benefit": _core},
        derived={"indexed_salary": lambda t, salary: salary * (1 + t * t)},
    )


def _assert_tree_close(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("limit", [None, 0.0, 0.375])
@pytest.mark.parametrize("record_every", [1, 4])
def test_scaled_matches_unfactored_for_all_kinds_and_views(limit, record_every):
    space = jact.StateSpace(["a", "b", "c"], [("a", "b"), ("b", "a"), ("b", "c")])
    model = space.build(transitions={edge: _hazard for edge in space.transitions})
    initial = space.initial_distribution(
        {
            "a": {"mass": 0.6, "duration": jnp.array([0.0, 0.1, 0.36, 0.0, 0.8])},
            "b": {"mass": 0.4, "duration": jnp.array([0.2, 0.0, 0.1, 0.7, 0.9])},
        }
    )
    views = {
        "raw": cf.Raw(),
        "one": cf.Raw("rate_income"),
        "group": cf.Group(["rate_income", "duration_expense"]),
        "total": cf.Total(),
        "state": cf.ByState(),
        "kind": cf.ByKind(),
        "pv": cf.Total(weight=lambda t, **kw: jnp.exp(-0.1 * t), terminal=True),
        "terminal_raw": cf.Raw(terminal=True),
    }
    kwargs: dict[str, Any] = dict(
        initial=initial,
        horizon=1,
        steps_per_unit=8,
        record_every=record_every,
        payment_duration_limit=limit,
        intensity_duration_limit=limit,
        cashflow_views=views,
        scale=jnp.linspace(0.8, 1.2, 5),
        salary=jnp.arange(1.0, 6.0),
        hazard=jnp.linspace(0.2, 0.5, 5),
        event_time=jnp.array([0.0, 0.36, 0.75, 1.0, -0.1]),
        target_duration=jnp.array([0.0, 0.36, 0.75, 1.0, -0.1]),
    )
    shared = model.solve(cashflows=_declaration(space, True), **kwargs)
    ordinary = model.solve(cashflows=_declaration(space, False), **kwargs)
    _assert_tree_close(shared, ordinary)
    for name, value in _cashflows(shared)["raw"].items():
        if name.endswith("zero"):
            np.testing.assert_array_equal(value, 0)
    np.testing.assert_allclose(
        _cashflows(shared)["total"],
        sum(_cashflows(shared)["raw"].values()),
        rtol=2e-6,
        atol=1e-6,
    )


def _prepare(space, components, cores):
    declaration = space.cashflows(components, cores=cores)
    # Only transition slots are needed for this structural check.
    matrix = tuple(
        tuple(
            _core if (source, target) in space.transitions else None
            for target in space.states
        )
        for source in space.states
    )
    return _prepare_cashflow_components(declaration, space.states, matrix)


def test_named_sharing_is_scoped_by_attachment_and_not_callable_identity():
    space = jact.StateSpace(["a", "b", "c"], [("a", "b"), ("a", "c")])

    def consumer(weight):
        return cf.StateRate({"a": cf.Scaled("x", weight=weight)})

    def first_when(**kw):
        return 0.5

    def second_when(**kw):
        return 0.5

    components = {
        "first": consumer(2),
        "second": consumer(3),
        "alias": cf.StateRate({"a": cf.Scaled("y")}),
        "other_state": cf.StateRate({"b": cf.Scaled("x")}),
        "ordinary": cf.StateRate({"a": _core}),
        "lump": cf.TransitionLump({("a", "b"): cf.Scaled("x")}),
        "other_edge": cf.TransitionLump({("a", "c"): cf.Scaled("x")}),
        "event1": cf.ScheduledEvent(first_when, {"a": cf.Scaled("x")}),
        "event2": cf.ScheduledEvent(first_when, {"a": cf.Scaled("x")}),
        "event3": cf.ScheduledEvent(second_when, {"a": cf.Scaled("x")}),
        "duration1": cf.DurationEvent({"a": 0.5}, {"a": cf.Scaled("x")}),
        "duration2": cf.DurationEvent({"a": jnp.array(0.5)}, {"a": cf.Scaled("x")}),
        "duration3": cf.DurationEvent({"a": 0.25}, {"a": cf.Scaled("x")}),
    }
    prepared = _prepare(space, components, {"x": _core, "y": _core})
    ids = [
        (c.targets if isinstance(c, DurationEventSpec) else c.payments)[0].task_id
        for c in prepared
    ]
    assert ids[0] == ids[1]
    assert ids[7] == ids[8]
    assert ids[10] == ids[11]
    assert len(set(ids)) == len(ids) - 3


def test_registry_is_frozen_unused_cores_are_not_called_and_unreachable_pruned():
    space = jact.StateSpace(["a", "unreachable"], [])

    def unused(t, d, **kw):
        raise AssertionError("Unused core must not be evaluated")

    def unit(t, d, **kw):
        return 1.0

    registry = {"unit": unit, "unused": unused}
    declaration = space.cashflows(
        {
            "a": cf.StateRate({"a": cf.Scaled("unit")}),
            "unreachable": cf.StateRate({"unreachable": cf.Scaled("unused")}),
        },
        cores=registry,
    )
    registry["unit"] = unused
    assert declaration.cores["unit"] is unit
    with pytest.raises(TypeError):
        cast(Any, declaration.cores)["unit"] = unused
    result = space.build(transitions={}).solve(
        "a",
        1,
        4,
        cashflows=declaration,
        cashflow_views={"pv": cf.Total(terminal=True)},
    )
    np.testing.assert_allclose(_cashflows(result)["pv"], 1)


@pytest.mark.parametrize(
    "payment,cores,error",
    [
        (cf.Scaled("missing"), {}, "unknown core"),
        (cf.Scaled(""), {}, "non-empty name"),
        (cf.Scaled(cast(Any, _core)), {}, "non-empty name"),
        (cf.Scaled(cast(Any, cf.Scaled("x"))), {}, "non-empty name"),
        (cf.Scaled("x"), {"x": cf.Scaled("x")}, "must be callable"),
        (cf.Scaled("x"), {"": _core}, "non-empty strings"),
        (cf.Scaled("x"), {"x": 3}, "must be callable"),
        (cf.Scaled("x", weight=jnp.ones(2)), {"x": _core}, "must be None"),
    ],
)
def test_invalid_declarations(payment, cores, error):
    space = jact.StateSpace(["a"], [])
    with pytest.raises((TypeError, ValueError), match=error):
        space.cashflows({"test": cf.StateRate({"a": payment})}, cores=cores)


@pytest.mark.parametrize("field", ["d", "duration_field", "indirect"])
def test_weights_cannot_read_duration_fields(field):
    space = jact.StateSpace(["a"], [])
    declaration = space.cashflows(
        {
            "test": cf.StateRate(
                {"a": cf.Scaled("x", weight=lambda t, **kw: kw[field])}
            ),
        },
        cores={"x": lambda t, d, **kw: 1.0},
        derived={
            "duration_field": lambda d: d,
            "indirect": lambda duration_field: duration_field + 1,
        },
    )
    with pytest.raises(ValueError, match="StateRate.*test.*unavailable field"):
        space.build(transitions={}).solve("a", 1, 4, cashflows=declaration)


def test_weight_shape_error_names_consumer():
    space = jact.StateSpace(["a"], [])
    declaration = space.cashflows(
        {
            "test": cf.StateRate(
                {
                    "a": cf.Scaled("x", weight=lambda t, **kw: jnp.ones((2, 3))),
                }
            ),
        },
        cores={"x": lambda t, d, **kw: 1.0},
    )
    with pytest.raises(ValueError, match="StateRate.*test.*weight"):
        space.build(transitions={}).solve("a", 1, 4, cashflows=declaration)


@dataclass
class _EqualCallable:
    value: float

    def __eq__(self, other):
        return isinstance(other, _EqualCallable)

    def __call__(self, t, d, **kw):
        return self.value


def test_same_name_different_definitions_do_not_reuse_stale_jit_code():
    space = jact.StateSpace(["a"], [])
    model = space.build(transitions={})
    for expected in (2.0, 7.0, 2.0):
        declaration = space.cashflows(
            {
                "test": cf.StateRate({"a": cf.Scaled("same_name")}),
            },
            cores={"same_name": _EqualCallable(expected)},
        )
        result = model.solve(
            "a",
            1,
            4,
            cashflows=declaration,
            cashflow_views={"pv": cf.Total(terminal=True)},
        )
        np.testing.assert_allclose(_cashflows(result)["pv"], expected)


def test_jit_and_gradients_match_unfactored_payments():
    space = jact.StateSpace(["a", "b"], [("a", "b")])
    model = space.build(transitions={("a", "b"): _hazard})

    def run(scale, salary, hazard, shared):
        def weight(t, **kw):
            return kw["salary"] * (1 + t)

        def ordinary(t, d, **kw):
            return _core(t, d, **kw) * weight(t, **kw)[:, None]

        declaration = space.cashflows(
            {
                "rate": cf.StateRate(
                    {
                        "a": cf.Scaled("x", weight=weight) if shared else ordinary,
                        "b": cf.Scaled("x", weight=weight) if shared else ordinary,
                    }
                ),
                "second_rate": cf.StateRate(
                    {
                        "a": cf.Scaled("x", weight=weight) if shared else ordinary,
                    }
                ),
                "lump": cf.TransitionLump(
                    {
                        ("a", "b"): cf.Scaled("x", weight=weight)
                        if shared
                        else ordinary,
                    }
                ),
            },
            cores={"x": _core},
        )
        result = model.solve(
            "a",
            1,
            4,
            initial_duration=0.2,
            cashflows=declaration,
            probability=None,
            cashflow_views={"pv": cf.Total(terminal=True)},
            scale=scale,
            salary=salary,
            hazard=hazard,
        )
        return jnp.sum(_cashflows(result)["pv"])

    inputs = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([0.1, 0.2]))
    shared = jax.jit(
        jax.value_and_grad(lambda a, b, c: run(a, b, c, True), argnums=(0, 1, 2))
    )
    ordinary = jax.jit(
        jax.value_and_grad(lambda a, b, c: run(a, b, c, False), argnums=(0, 1, 2))
    )
    _assert_tree_close(shared(*inputs), ordinary(*inputs))


def _count_primitive(value, name):
    if hasattr(value, "eqns"):
        return sum(
            (eq.primitive.name == name) + _count_primitive(eq.params, name)
            for eq in value.eqns
        )
    if hasattr(value, "jaxpr"):
        return _count_primitive(value.jaxpr, name)
    if isinstance(value, dict):
        return sum(_count_primitive(v, name) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_count_primitive(v, name) for v in value)
    return 0


@pytest.mark.parametrize("kind", ["rate", "lump", "scheduled", "duration"])
def test_traced_program_evaluates_each_named_core_once(kind):
    space = jact.StateSpace(["a", "b", "c"], [("a", "b"), ("b", "c")])
    model = space.build(transitions={edge: _hazard for edge in space.transitions})

    def count(names):
        declaration = space.cashflows(
            {
                str(i): _component(kind, cf.Scaled(name, weight=i + 2))
                for i, name in enumerate(names)
            },
            cores={"x": _core, "y": _core},
        )
        traced = jax.make_jaxpr(
            lambda scale: model.solve(
                "a",
                1,
                4,
                cashflows=declaration,
                probability=None,
                scale=scale,
                hazard=jnp.ones(2),
                event_time=0.5,
                target_duration=0.5,
            )
        )(jnp.ones(2))
        # Duration events sample a target, whereas the other kinds reduce a grid.
        primitive = "exp" if kind == "duration" else "reduce_sum"
        return _count_primitive(traced, primitive)

    single, shared, separate = count(["x"]), count(["x", "x"]), count(["x", "y"])
    assert shared == single
    assert separate > shared


@pytest.mark.parametrize("weight,expected", [(None, 1.0), (jnp.array(2.5), 2.5)])
def test_scalar_weights_inside_outer_jit(weight, expected):
    space = jact.StateSpace(["a"], [])
    model = space.build(transitions={})
    declaration = space.cashflows(
        {
            "payment": cf.StateRate({"a": cf.Scaled("unit", weight=weight)}),
        },
        cores={"unit": lambda t, d, **kw: 1.0},
    )
    result = jax.jit(
        lambda: model.solve(
            "a",
            1,
            4,
            cashflows=declaration,
            cashflow_views={"pv": cf.Total(terminal=True)},
        )
    )()
    np.testing.assert_allclose(_cashflows(result)["pv"], expected)
