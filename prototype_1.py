import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Sequence, Any, Dict, Optional, Union
from probability_callbacks import ProbabilityCallbacks

@jax.jit
def roll_probability_tensor(p: jnp.ndarray):
    p = jnp.roll(p, shift=1, axis=-1)
    p = p.at[..., 0].set(0)

    return p


@jax.jit
def update_p(p: jnp.ndarray, delta: jnp.ndarray, step_size: jnp.ndarray):
    step_size = jnp.expand_dims(step_size, axis=-1)
    p = p + step_size * delta
    p = roll_probability_tensor(p)
    return p

@jax.jit
def update_p_point(p: jnp.ndarray, delta: jnp.ndarray, step_size: jnp.ndarray):
    p = p + step_size * delta
    p = roll_probability_tensor(p)
    return p


@jax.jit
def compute_derivative(
    p: jnp.ndarray, p_point: jnp.ndarray, mu_plus: jnp.ndarray, mu_minus: jnp.ndarray
):

    outflow_plus, inflow_plus = mu_plus
    outflow_minus, inflow_minus = mu_minus

    outflow_plus = jnp.sum(outflow_plus, axis=-2)
    outflow_minus = jnp.sum(outflow_minus, axis=-2)

    dp = jnp.diff(p, axis=-1)
    dp_point = jnp.diff(p_point, axis=-1, prepend=0)

    outflow_avg = 0.5 * (outflow_plus[..., :-1] + outflow_minus[..., 1:])
    inflow_avg = 0.5 * (inflow_plus[..., :-1] + inflow_minus[..., 1:])

    outflow_integral = jnp.cumulative_sum(
        outflow_avg * dp, axis=-1, include_initial=1
    )  # B X J X D
    inflow_integral = jnp.sum(
        inflow_avg * jnp.expand_dims(dp, axis=-3), axis=(-2, -1)
    )  # B X J
    inflow_integral = jnp.expand_dims(inflow_integral, axis=-1)

    outflow_point_integral = jnp.cumsum(outflow_plus[..., 0, :] * dp_point, axis=-1)

    inflow_point_integral = inflow_plus[..., 0, :] * jnp.expand_dims(dp_point, axis=-2)
    inflow_point_integral = jnp.sum(inflow_point_integral, axis=-1, keepdims=True)

    inflow_integral = inflow_integral + inflow_point_integral

    delta_p = inflow_integral - outflow_integral
    delta_p_point = -outflow_point_integral

    return delta_p, delta_p_point


@jax.jit
def step_sizes_from_grid(grid: jnp.ndarray) -> jnp.ndarray:
    """Calculates step sizes from a solution grid

    The semi-markov solver takes a solution grid as input from which
    step sizes for the solver is dervived.

    Parameters
    ----------
    grid : jnp.ndarray
          A 2D array representing the grid points of the solution grid.

    Returns
    -------
    jnp.ndarray
          A 3D array containing the step sizes needed for the semi-markov solver.
    """
    step_sizes = jnp.diff(grid, axis=-1)
    step_sizes = jnp.swapaxes(step_sizes, 0, 1)
    step_sizes = jnp.expand_dims(step_sizes, axis=-1)

    return step_sizes

@jax.jit
def concatenate_init_probability(x, x0):
    return jnp.concatenate([jnp.expand_dims(x0, axis=0), x])


@partial(
    jax.jit,
    static_argnames=['intensity', 'prob_callback'],
)
def heun_scheme_solver(
    p_0: jnp.ndarray,
    p_point_0: jnp.ndarray,
    grid: jnp.ndarray,
    intensity: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    prob_callback: Callable[..., jnp.ndarray],
    pertubation: jnp.ndarray,
):

    grid_minus = grid - pertubation
    grid_plus = grid + pertubation

    step_sizes = step_sizes_from_grid(grid)
    
    def heun_scan(carry, step_size):
        p, p_point, t = carry

        t_left = t + pertubation

        mu_plus = intensity(t_left, grid_plus, **intensity_kwargs)
        mu_minus = intensity(t_left, grid_minus, **intensity_kwargs)

        delta_p, delta_p_point = compute_derivative(p, p_point, mu_plus, mu_minus)

        t += step_size
        
        p_2 = update_p(p, delta_p, step_size)
        
        p_point_2 = update_p_point(p_point, delta_p_point, step_size)
        
        """
        t_right = t - pertubation

        mu_plus = intensity(t_right, grid_plus, **intensity_kwargs)
        mu_minus = intensity(t_right, grid_minus, **intensity_kwargs)

        delta_p_2, delta_p_point_2 = compute_derivative(
            p_2, p_point_2, mu_plus, mu_minus
        )

        delta_p_2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
        delta_p_point2 = 0.5 * (delta_p_point_2[..., 1:] + delta_p_point[..., :-1])

        delta_p = delta_p.at[..., :-1].set(delta_p_2)
        delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)
        
        p = update_p(p, delta_p, step_size)
        p_point = update_p_point(p_point, delta_p_point, step_size)
        """

        #next_carry = (p, p_point, t)
        next_carry = (p_2, p_point_2, t)
        history = {
            'probability': prob_callback(p, p_point),
        }

        return next_carry, history

    t0 = jnp.zeros_like(step_sizes[0])

    _, result = jax.lax.scan(heun_scan, (p_0, p_point_0, t0), step_sizes)
       
    init_prob_callback_value = prob_callback(p_0, p_point_0)
    result['probability'] = jax.tree_util.tree_map(
        lambda arr, init: concatenate_init_probability(arr, init),
        result['probability'],
        init_prob_callback_value
    )

    return result


@jax.jit
def transpose_probability(x):
    N = x.ndim
    if N == 1:
        return x
    if N == 2:
        return jnp.transpose(x, axes=(1, 0))

    return jnp.moveaxis(x, 0, -2)

@partial(
    jax.jit,
    static_argnames=['intensity', 'prob_callback', 'transpose_result'],
)
def semimarkov_solver(
    grid: jnp.ndarray,
    intensity: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    intensity_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    prob_callback: Union[None, str, Callable[..., Any]] = 'default',
    pertubation: jnp.ndarray = 1e-12,
    transpose_result: bool = True,
):

    if not callable(prob_callback):
        prob_callback = ProbabilityCallbacks.from_str(prob_callback)
        
    intensity_kwargs = {} if intensity_kwargs is None else intensity_kwargs

    outflow, inflow = intensity(0, grid, **intensity_kwargs)

    p_point_0 = jnp.ones_like(outflow[..., 0, 0, :])
    p_0 = jnp.zeros_like(outflow[..., 0, :])

    result = heun_scheme_solver(
        p_0,
        p_point_0,
        grid,
        intensity,
        intensity_kwargs,
        prob_callback,
        pertubation,
    )

    if transpose_result:
        transposed_prob = jax.tree_util.tree_map(
            transpose_probability, result["probability"]
        )
        result["probability"] = transposed_prob

    return result
