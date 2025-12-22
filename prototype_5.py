import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Sequence, Any, Dict, Optional, Union
from probability_callbacks import ProbabilityCallbacks
from function_utils import get_reference_function

@jax.jit
def roll_probability_tensor(p: jnp.ndarray, next_inflow: jnp.ndarray):
    p = jnp.roll(p, shift=1, axis=-1)
    p = p.at[..., 0].set(next_inflow)

    return p

@jax.jit
def update_p(p: jnp.ndarray, delta: jnp.ndarray, next_inflow: jnp.ndarray, step_size: float):
    p = p + step_size * delta
    p = roll_probability_tensor(p, step_size * next_inflow)
    return p

@jax.jit
def update_p_point(p: jnp.ndarray, delta: jnp.ndarray, step_size: jnp.ndarray):
    p = p + step_size * delta
    p = roll_probability_tensor(p, 0)
    return p


@jax.jit
def compute_derivative(
    p: jnp.ndarray, p_point: jnp.ndarray, mu_plus_matrix, mu_minus_matrix
):
    B, J, D_minus_1 = p.shape
    
    outflow_plus = jnp.zeros((B, J, D_minus_1))
    outflow_avg = jnp.zeros((B, J, D_minus_1))
    next_inflow = jnp.zeros((B, J))
    
    for i in range(J):
        for j in range(J):
            m_p = mu_plus_matrix[i][j]
            m_m = mu_minus_matrix[i][j]
            
            if m_p is not None and m_m is not None:
                m_p_slice = m_p[:, :-1]
                m_avg = 0.5 * (m_p_slice + m_m[:, 1:])

                outflow_plus = outflow_plus.at[:, i, :].add(m_p_slice)
                outflow_avg = outflow_avg.at[:, i, :].add(m_avg)

                inflow_integral = jnp.sum(m_avg * p[:, i, :], axis=-1)
                
                if i == 0:
                    inflow_point_integral = jnp.sum(m_p_slice * p_point, axis=-1)
                    next_inflow = next_inflow.at[:, j].add(inflow_point_integral)
                
                next_inflow = next_inflow.at[:, j].add(inflow_integral)

    delta_p = -p * outflow_avg
    delta_p_point = -outflow_plus[:, 0, :] * p_point 

    return next_inflow, delta_p, delta_p_point

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

def evaluate_functions(matrix, *args, **kwargs):
    output = jax.tree_util.tree_map(
        lambda f: f(*args, **kwargs),
        matrix
    )
    return output

@partial(
    jax.jit,
    static_argnames=['step_size', 'intensity', 'prob_callback', 'pertubation'],
)
def heun_scheme_solver(
    p_0: jnp.ndarray,
    p_point_0: jnp.ndarray,
    grid: jnp.ndarray,
    step_size: float,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Dict[str, jnp.ndarray],
    prob_callback: Callable[..., jnp.ndarray],
    pertubation: jnp.ndarray,
):

    grid_minus = grid - pertubation
    grid_plus = grid + pertubation
    
    def heun_scan(carry, t):
        p, p_point = carry

        t_left = t + pertubation
        
        mu_plus = evaluate_functions(intensity, t_left, grid_plus, **intensity_kwargs)
        mu_minus = evaluate_functions(intensity, t_left, grid_minus, **intensity_kwargs)
        
        next_inflow, delta_p, delta_p_point = compute_derivative(p, p_point, mu_plus, mu_minus)

        t += step_size
        
        p_2 = update_p(p, delta_p, next_inflow, step_size)
        p_point_2 = update_p_point(p_point, delta_p_point, step_size)
        
        t_right = t - pertubation
        
        mu_plus = evaluate_functions(intensity, t_right, grid_plus, **intensity_kwargs)
        mu_minus = evaluate_functions(intensity, t_right, grid_minus, **intensity_kwargs)
        
        next_inflow_2, delta_p_2, delta_p_point_2 = compute_derivative(
            p_2, p_point_2, mu_plus, mu_minus
        )
        
        next_inflow_2 = 0.5 * (next_inflow + next_inflow_2 + delta_p_2[..., 0])
        delta_p2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
        delta_p_point2 = 0.5 * (delta_p_point_2[..., 1:] + delta_p_point[..., :-1])

        delta_p = delta_p.at[..., :-1].set(delta_p2)
        delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)
        
        p = update_p(p, delta_p, next_inflow_2, step_size)
        p_point = update_p_point(p_point, delta_p_point, step_size)
        
        next_carry = (p, p_point)
        
        history = {
            'probability': prob_callback(p, p_point),
        }

        return next_carry, history
    
    scan_grid = jnp.swapaxes(grid[..., :-1], 0, -1)

    _, result = jax.lax.scan(heun_scan, (p_0, p_point_0), scan_grid)
       
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
    static_argnames=['units', 'discretization_unit', 'intensity', 'prob_callback', 'transpose_result'],
)
def semimarkov_solver(
    units: int,
    discretization_unit: int,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    prob_callback: Union[None, str, Callable[..., Any]] = 'default',
    pertubation: jnp.ndarray = 1e-12,
    transpose_result: bool = True,
):
    n_states = len(intensity)
    solver_steps = discretization_unit * units
    grid = jnp.linspace(
        0, units, solver_steps + 1, endpoint=True 
    )
    grid = jnp.expand_dims(grid, 0)
    step_size = 1 / discretization_unit
    
    if not callable(prob_callback):
        prob_callback = ProbabilityCallbacks.from_str(prob_callback)
        
    intensity_kwargs = {} if intensity_kwargs is None else intensity_kwargs
    
    reference_function = get_reference_function(intensity)
    dummy = reference_function(0, grid, **intensity_kwargs)
    batch_size = dummy.shape[0]
    
    p_point_0 = jnp.zeros((batch_size, solver_steps))
    p_0 = jnp.zeros((batch_size, n_states, solver_steps))
    p_point_0 = p_point_0.at[..., 0].set(1)

    result = heun_scheme_solver(
        p_0,
        p_point_0,
        grid,
        step_size,
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
