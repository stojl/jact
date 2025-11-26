import jax
import jax.numpy as jnp


@jax.jit
def prob_callback_none(p: jnp.ndarray, p_point: jnp.ndarray) -> None:
    return None


@jax.jit
def prob_callback_default(
    p: jnp.ndarray, p_point: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return p, p_point


@jax.jit
def prob_callback_no_duration(
    p: jnp.ndarray, p_point: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return p[..., -1], p_point[..., -1]


@jax.jit
def prob_callback_collapse_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
    p_with_point = p[..., 0, :] + jnp.expand_dim(p_point, axis=1)
    p = p.at[..., 0, :].set(p_with_point)
    return p


@jax.jit
def prob_callback_collapse_point_no_duration(
    p: jnp.ndarray, p_point: jnp.ndarray
) -> jnp.ndarray:
    p = prob_callback_collapse_point(p, p_point)
    return p[..., -1]


@jax.jit
def prob_callback_point_only(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
    return p_point


@jax.jit
def prob_callback_point_only_no_duration(
    p: jnp.ndarray, p_point: jnp.ndarray
) -> jnp.ndarray:
    return p_point[..., -1]


@jax.jit
def prob_callback_no_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
    return p


@jax.jit
def prob_callback_no_point_no_duration(
    p: jnp.ndarray, p_point: jnp.ndarray
) -> jnp.ndarray:
    return p[..., -1]


@jax.jit
def get_probability_callback_from_str(str: str):
    match str:
        case "default":
            return prob_callback_default
        case "none":
            return prob_callback_none
        case "no_duration":
            return prob_callback_no_duration
        case "collapse_point":
            return prob_callback_collapse_point
        case "collapse_point_no_duration":
            return prob_callback_collapse_point_no_duration
        case "point_only":
            return prob_callback_point_only
        case "point_only_no_duration":
            return prob_callback_point_only_no_duration
        case "no_point":
            return prob_callback_no_point
        case "no_point_no_duration":
            return prob_callback_no_point_no_duration
        case _:
            return prob_callback_default
