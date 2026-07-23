# pyright: strict, reportMissingImports=false, reportUntypedClassDecorator=false
"""Typed result of :meth:`jact.Model.simulate`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["SimulationResult"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationResult:
    """Fixed-shape event histories returned by ``Model.simulate``.

    The leading axes of every array are ``(individual, replicate)``.
    Unused jump slots contain ``NaN`` and unused state-path slots contain
    ``-1``.
    """

    states: tuple[str, ...]
    jump_times: jax.Array
    jump_durations: jax.Array
    state_path: jax.Array
    jump_count: jax.Array
    overflow: jax.Array
    truncated_at_time: jax.Array
    final_state: jax.Array
    final_duration: jax.Array

    def tree_flatten(
        self,
    ) -> tuple[tuple[jax.Array, ...], tuple[str, ...]]:
        children = (
            self.jump_times,
            self.jump_durations,
            self.state_path,
            self.jump_count,
            self.overflow,
            self.truncated_at_time,
            self.final_state,
            self.final_duration,
        )
        return children, self.states

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[str, ...],
        children: tuple[jax.Array, ...],
    ) -> SimulationResult:
        return cls(aux, *children)

    def valid_mask(self) -> jax.Array:
        """Return the boolean mask of populated jump-buffer entries."""
        return (
            jnp.arange(self.jump_times.shape[-1], dtype=self.jump_count.dtype)
            < self.jump_count[..., None]
        )

    def path(self, individual: int, replicate: int = 0) -> dict[str, Any]:
        """Extract one trajectory as compact host-side arrays and state names."""
        batch_size, replicates = self.jump_count.shape
        if not 0 <= individual < batch_size:
            raise IndexError(
                f"individual must be in [0, {batch_size}), got {individual}."
            )
        if not 0 <= replicate < replicates:
            raise IndexError(
                f"replicate must be in [0, {replicates}), got {replicate}."
            )

        count = int(np.asarray(self.jump_count[individual, replicate]))
        state_indices = np.asarray(
            self.state_path[individual, replicate, : count + 1]
        )
        return {
            "jump_times": np.asarray(
                self.jump_times[individual, replicate, :count]
            ),
            "jump_durations": np.asarray(
                self.jump_durations[individual, replicate, :count]
            ),
            "state_indices": state_indices,
            "state_names": tuple(self.states[int(index)] for index in state_indices),
            "overflow": bool(np.asarray(self.overflow[individual, replicate])),
            "truncated_at_time": float(
                np.asarray(self.truncated_at_time[individual, replicate])
            ),
            "final_state": int(
                np.asarray(self.final_state[individual, replicate])
            ),
            "final_duration": float(
                np.asarray(self.final_duration[individual, replicate])
            ),
        }

    def to_pandas(self) -> Any:
        """Convert populated jumps to a pandas ``DataFrame``.

        Pandas is optional and imported only when this method is called.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SimulationResult.to_pandas() requires pandas."
            ) from exc

        counts = np.asarray(self.jump_count)
        times = np.asarray(self.jump_times)
        durations = np.asarray(self.jump_durations)
        paths = np.asarray(self.state_path)
        rows: list[dict[str, Any]] = []
        for individual in range(counts.shape[0]):
            for replicate in range(counts.shape[1]):
                for jump_number in range(int(counts[individual, replicate])):
                    source = int(paths[individual, replicate, jump_number])
                    target = int(paths[individual, replicate, jump_number + 1])
                    rows.append(
                        {
                            "individual": individual,
                            "replicate": replicate,
                            "jump_number": jump_number,
                            "jump_time": float(
                                times[individual, replicate, jump_number]
                            ),
                            "jump_duration": float(
                                durations[individual, replicate, jump_number]
                            ),
                            "from_state": self.states[source],
                            "to_state": self.states[target],
                        }
                    )
        return pd.DataFrame.from_records(
            rows,
            columns=[
                "individual",
                "replicate",
                "jump_number",
                "jump_time",
                "jump_duration",
                "from_state",
                "to_state",
            ],
        )

