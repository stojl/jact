"""Unit tests for jact.InitialDistribution JIT boundary, per docs/api_spec.md.

These tests lock in the JIT-boundary contract documented in
`docs/api_spec.md` §InitialDistribution and §Solver → JIT boundary:

- `mass` and `duration` arrays are *traced* (runtime values).
- The declared initial-state set is *static* (trace-time constant),
  driven by the keys of `components`, the state passed to `at`, or the
  `initial_states` tuple on `per_individual` — never by mass values or
  by the contents of an index array.
- `per_individual` is trace-clean: it may be called from inside the
  user's own ``jax.jit`` / ``vmap``.

Group A covers trace-cleanness of the constructors; Group B covers the
static-vs-traced separation via pytree structure equality.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import jact


BATCH = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tree_struct(obj):
    return jax.tree_util.tree_structure(obj)


@pytest.fixture(scope="module")
def _pytree_registered():
    """Skip Group B if `InitialDistribution` is not a registered JAX pytree.

    When an object is not a registered pytree, JAX treats it as an opaque
    leaf and `tree_structure` collapses to a single-leaf structure that
    cannot distinguish between instances. The spec's JIT-boundary claims
    are only meaningful when the object is a pytree; we surface this as a
    skip rather than a silent pass.
    """
    dist = jact.InitialDistribution.at("healthy", duration=jnp.array(0.0))
    struct = _tree_struct(dist)
    if struct.num_leaves == 0 or (
        struct.num_leaves == 1 and _tree_struct(jnp.array(0.0)) == struct
    ):
        pytest.skip("InitialDistribution is not registered as a JAX pytree")


# ---------------------------------------------------------------------------
# Group A — trace-cleanness of constructors
# ---------------------------------------------------------------------------


class TestTraceCleanConstructors:
    def test_at_accepts_traced_scalar_duration(self):
        """`InitialDistribution.at` survives tracing with a traced duration."""

        def build(d):
            return jact.InitialDistribution.at("healthy", duration=d)

        dist = jax.jit(build)(jnp.array(0.5))
        assert isinstance(dist, jact.InitialDistribution)

    def test_at_accepts_traced_batch_duration(self):
        """`at` accepts a (batch,) traced duration (per-individual d_0)."""

        def build(d):
            return jact.InitialDistribution.at("healthy", duration=d)

        dist = jax.jit(build)(jnp.zeros((BATCH,)))
        assert isinstance(dist, jact.InitialDistribution)

    def test_per_individual_accepts_traced_states_and_duration(self):
        """Spec: `states` is a traced (batch,) int32 array and
        `per_individual` may be called from inside the user's own jit."""

        def build(states, duration):
            return jact.InitialDistribution.per_individual(
                states=states,
                duration=duration,
                initial_states=("healthy", "disabled"),
            )

        states = jnp.zeros((BATCH,), dtype=jnp.int32)
        duration = jnp.zeros((BATCH,))
        dist = jax.jit(build)(states, duration)
        assert isinstance(dist, jact.InitialDistribution)

    def test_primary_constructor_accepts_traced_mass_and_duration(self):
        """Primary constructor trace-clean with traced mass/duration.

        `normalise=False` isolates the trace-cleanness question from the
        sum-to-1 validation, which is a separate concern (see plan).
        """

        def build(m, d):
            return jact.InitialDistribution(
                components={"healthy": {"mass": m, "duration": d}},
                normalise=False,
            )

        mass = jnp.ones((BATCH,))
        duration = jnp.zeros((BATCH,))
        dist = jax.jit(build)(mass, duration)
        assert isinstance(dist, jact.InitialDistribution)


# ---------------------------------------------------------------------------
# Group B — static set, traced values
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_pytree_registered")
class TestStaticVsTracedSeparation:
    def test_same_keys_different_values_same_tree_structure(self):
        """Changing mass/duration values with the same declared set must
        not change the pytree structure (⇒ no downstream retrace)."""
        a = jact.InitialDistribution(
            components={
                "healthy": {"mass": jnp.ones((BATCH,)), "duration": jnp.zeros((BATCH,))},
            },
            normalise=True,
        )
        b = jact.InitialDistribution(
            components={
                "healthy": {
                    "mass": jnp.ones((BATCH,)),
                    "duration": jnp.full((BATCH,), 0.37),
                },
            },
            normalise=True,
        )
        assert _tree_struct(a) == _tree_struct(b)

    def test_different_keys_different_tree_structure(self):
        """Changing the declared initial-state set *must* change the
        pytree structure so a downstream jitted consumer retraces."""
        a = jact.InitialDistribution(
            components={
                "healthy": {"mass": jnp.ones((BATCH,)), "duration": jnp.zeros((BATCH,))},
            },
            normalise=True,
        )
        b = jact.InitialDistribution(
            components={
                "healthy": {
                    "mass": jnp.full((BATCH,), 0.5),
                    "duration": jnp.zeros((BATCH,)),
                },
                "disabled": {
                    "mass": jnp.full((BATCH,), 0.5),
                    "duration": jnp.zeros((BATCH,)),
                },
            },
            normalise=True,
        )
        assert _tree_struct(a) != _tree_struct(b)

    def test_zero_mass_component_still_allocates_a_slot(self):
        """Per §Static-topology invariant: a component declared with
        all-zero mass still allocates a point-mass slot. The declared set
        is driven by keys, not by values."""
        dist_zero = jact.InitialDistribution(
            components={
                "healthy": {"mass": jnp.ones((BATCH,)), "duration": jnp.zeros((BATCH,))},
                "disabled": {
                    "mass": jnp.zeros((BATCH,)),
                    "duration": jnp.zeros((BATCH,)),
                },
            },
            normalise=True,
        )
        dist_half = jact.InitialDistribution(
            components={
                "healthy": {
                    "mass": jnp.full((BATCH,), 0.5),
                    "duration": jnp.zeros((BATCH,)),
                },
                "disabled": {
                    "mass": jnp.full((BATCH,), 0.5),
                    "duration": jnp.zeros((BATCH,)),
                },
            },
            normalise=True,
        )
        assert _tree_struct(dist_zero) == _tree_struct(dist_half)

    def test_per_individual_states_values_do_not_affect_structure(self):
        """`states` index values are traced; only `initial_states` drives
        the static set."""
        a = jact.InitialDistribution.per_individual(
            states=jnp.zeros((BATCH,), dtype=jnp.int32),
            duration=jnp.zeros((BATCH,)),
            initial_states=("healthy", "disabled"),
        )
        b = jact.InitialDistribution.per_individual(
            states=jnp.arange(BATCH, dtype=jnp.int32) % 2,
            duration=jnp.zeros((BATCH,)),
            initial_states=("healthy", "disabled"),
        )
        assert _tree_struct(a) == _tree_struct(b)

    def test_per_individual_initial_states_change_retraces(self):
        """Changing the `initial_states` tuple changes the declared set
        and must show up as a pytree-structure difference."""
        a = jact.InitialDistribution.per_individual(
            states=jnp.zeros((BATCH,), dtype=jnp.int32),
            duration=jnp.zeros((BATCH,)),
            initial_states=("healthy",),
        )
        b = jact.InitialDistribution.per_individual(
            states=jnp.zeros((BATCH,), dtype=jnp.int32),
            duration=jnp.zeros((BATCH,)),
            initial_states=("healthy", "disabled"),
        )
        assert _tree_struct(a) != _tree_struct(b)
