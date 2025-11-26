import jax
import jax.numpy as jnp


class ProbabilityCallbacks:

    @staticmethod
    @jax.jit
    def empty(p: jnp.ndarray, p_point: jnp.ndarray) -> None:
        return None

    @staticmethod
    @jax.jit
    def default(
        p: jnp.ndarray, p_point: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return p, p_point

    @staticmethod
    @jax.jit
    def no_duration(
        p: jnp.ndarray, p_point: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return p[..., -1], p_point[..., -1]

    @staticmethod
    @jax.jit
    def collapse_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        p_with_point = p[..., 0, :] + jnp.expand_dim(p_point, axis=1)
        p = p.at[..., 0, :].set(p_with_point)
        return p

    @staticmethod
    @jax.jit
    def collapse_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        p = ProbabilityCallbacks.collapse_point(p, p_point)
        return p[..., -1]

    @staticmethod
    @jax.jit
    def point_only(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        return p_point

    @staticmethod
    @jax.jit
    def point_only_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        return p_point[..., -1]

    @staticmethod
    @jax.jit
    def no_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        return p

    @staticmethod
    @jax.jit
    def no_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        return p[..., -1]

    @staticmethod
    def get_probability_callback_from_str(str: str):
        match str:
            case "default":
                return ProbabilityCallbacks.default
            case "none":
                return ProbabilityCallbacks.empty
            case "no_duration":
                return ProbabilityCallbacks.no_duration
            case "collapse_point":
                return ProbabilityCallbacks.collapse_point
            case "collapse_point_no_duration":
                return ProbabilityCallbacks.collapse_point_no_duration
            case "point_only":
                return ProbabilityCallbacks.point_only
            case "point_only_no_duration":
                return ProbabilityCallbacks.point_only_no_duration
            case "no_point":
                return ProbabilityCallbacks.no_point
            case "no_point_no_duration":
                return ProbabilityCallbacks.no_point_no_duration
            case _:
                return ProbabilityCallbacks.default
