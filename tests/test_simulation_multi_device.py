"""Forced-CPU multi-device checks in a clean JAX subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_multi_device_simulation_is_exact_and_padding_safe():
    project_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        import jact

        devices = jax.local_devices()
        assert len(devices) == 4

        state_space = jact.StateSpace(
            states=("a", "b"),
            transitions=(("a", "b"), ("b", "a")),
        )

        def a_to_b(t, d, *, exposure, scale):
            del t
            return scale / exposure[:, None] + 0.1 * d

        def b_to_a(t, d, *, exposure, scale):
            del t
            return 0.5 * scale / exposure[:, None] + 0.05 * d

        model = state_space.build(
            transitions={
                ("a", "b"): a_to_b,
                ("b", "a"): b_to_a,
            }
        )
        initial = jact.InitialDistribution(
            components={
                "a": {
                    "mass": jnp.asarray([0.2, 0.4, 0.6, 0.8, 1.0]),
                    "duration": jnp.asarray([0.0, 0.1, 0.2, 0.3, 0.4]),
                },
                "b": {
                    "mass": jnp.asarray([0.8, 0.6, 0.4, 0.2, 0.0]),
                    "duration": jnp.asarray([0.5, 0.4, 0.3, 0.2, 0.1]),
                },
            }
        )
        exposure = jnp.asarray([1.0, 1.5, 2.0, 2.5, 3.0])

        def run(key, selected):
            return model.simulate(
                initial=initial,
                horizon=2,
                steps_per_unit=4,
                max_jumps=12,
                replicates=4,
                key=key,
                devices=selected,
                exposure=exposure,
                scale=0.7,
            )

        def assert_equal(left, right):
            assert left.states == right.states
            for left_leaf, right_leaf in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
                strict=True,
            ):
                np.testing.assert_array_equal(left_leaf, right_leaf)

        for key in (jax.random.key(123), jax.random.PRNGKey(123)):
            reference = run(key, None)
            for selected in (1, 2, 4, tuple(devices[1:3])):
                candidate = run(key, selected)
                assert candidate.jump_times.shape == (5, 4, 12)
                assert_equal(reference, candidate)

        grouped_space = jact.StateSpace(
            states=("active", "cause_a", "cause_b"),
            transitions=(
                ("active", "cause_a"),
                ("active", "cause_b"),
            ),
        )

        def grouped_exits(t, d, *, exposure):
            del t
            base = 1.0 / exposure[:, None] + 0.0 * d
            return jnp.stack((base, 2.0 * base), axis=0)

        grouped_model = grouped_space.build(exits={"active": grouped_exits})
        grouped_arguments = dict(
            initial="active",
            horizon=2,
            steps_per_unit=2,
            max_jumps=1,
            replicates=3,
            key=jax.random.key(77),
            exposure=exposure,
        )
        grouped_reference = grouped_model.simulate(
            **grouped_arguments,
            devices=None,
        )
        grouped_mapped = grouped_model.simulate(
            **grouped_arguments,
            devices=4,
        )
        assert_equal(grouped_reference, grouped_mapped)

        overflow_model = state_space.build(
            transitions={
                ("a", "b"): lambda t, d, **kwargs: 100.0,
                ("b", "a"): lambda t, d, **kwargs: 100.0,
            }
        )
        try:
            overflow_model.simulate(
                initial="a",
                horizon=1,
                steps_per_unit=1,
                max_jumps=0,
                replicates=2,
                key=jax.random.key(9),
                devices=4,
                overflow="raise",
            )
        except RuntimeError as exc:
            assert "2 of 2 trajectories" in str(exc)
        else:
            raise AssertionError("Expected overflow='raise' to fail.")
        """
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
    )
