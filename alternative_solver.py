import jax
import jax.numpy as jnp
from typing import Callable, Sequence, Any
from functools import partial

def _get_zero_placeholder(func: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    """Returns a new function that returns zeros of the same shape as func."""
    def zero_placeholder_func(*args, **kwargs):
        return func(*args, **kwargs) * 0.0

    return zero_placeholder_func

def fill_intensity_matrix(
        intensity_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]],
        fill_func = Callable[..., jnp.ndarray]
) -> Sequence[Sequence[Callable[..., jnp.ndarray]]]:
    filled_matrix = [
        [
            (func if func is not None else fill_func)
            for func in row 
        ]
        for row in intensity_matrix
    ]
    
    return filled_matrix

def construct_func_from_intensity_matrix(
        intensity_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]]
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Returns a new function that computes the outflow and inflow according to an
    n x n intensity matrix.
    """
    n = len(intensity_matrix)
    if not all(len(row) == n for row in intensity_matrix):
        raise ValueError("Intensity matrix must be square (n x n).")
    
    all_functions = (f for row in intensity_matrix for f in row)
    global_reference_func = next((f for f in all_functions if f is not None), None)
    
    if global_reference_func is None:
        raise ValueError("The intensity matrix contains only None entries. Cannot construct any function or determine output shape.")
    
    zero_func = _get_zero_placeholder(global_reference_func)
    intensity_matrix = fill_intensity_matrix(intensity_matrix, zero_func)
    
    @jax.jit
    def outflow_inflow(*args: Any, **kwargs: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        evaluated_matrix = [
            [func(*args, **kwargs)for func in row]
            for row in intensity_matrix
        ]

        outflow = [sum(row) for row in evaluated_matrix]
        outflow = jnp.stack(outflow, axis=-2)

        tranposed_matrix = list(zip(*evaluated_matrix))
        inflow = [jnp.stack(col, axis=-2) for col in tranposed_matrix]
        inflow = jnp.stack(inflow, axis=-3)       

        return outflow, inflow
    
    return outflow_inflow

@jax.jit
def next_p_j(p_j, outflow, inflow, step_size):
    dp_j = jnp.diff(p_j, axis=-1, prepend=0)
    outflow_integral = jnp.cumsum(outflow * dp_j, axis=-1)
    inflow_integral = jnp.sum(inflow * jnp.expand_dims(dp_j, axis=-3), axis=(-2,-1))
    inflow_integral = jnp.expand_dims(inflow_integral, axis=-1)
    
    p_j = p_j + jnp.expand_dims(step_size, axis=-1) * (inflow_integral - outflow_integral)
    p_j = jnp.roll(p_j, shift=1, axis=-1)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j

@partial(jax.jit, static_argnames=['flow'])
def solve_p_j(p_j_0: jnp.ndarray, 
              step_sizes: jnp.ndarray, 
              flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]], 
              *args: jnp.ndarray,
              **kwargs: jnp.ndarray):

    def scan_p_j(carry, step_size):
        p_j, t = carry
        outflow, inflow = flow(t, *args, **kwargs)
        p_j = next_p_j(p_j, outflow, inflow, step_size)
        t += step_size

        return (p_j, t), p_j
    
    t0 = jnp.zeros_like(step_sizes[0])
    _, p_j = jax.lax.scan(scan_p_j, (p_j_0, t0), step_sizes)
    
    p_j = jnp.concatenate([jnp.expand_dims(p_j_0, axis=0), p_j])
    p_j = jnp.swapaxes(p_j, 0, 1)
    p_j = jnp.swapaxes(p_j, 1, 2)

    return p_j

@jax.jit
def next_p_j_heun(p_j, outflow, inflow, step_size):
    dp_j = jnp.diff(p_j, axis=-1, prepend=0)
    outflow_integral = jnp.cumsum(outflow * dp_j, axis=-1)
    inflow_integral = jnp.sum(inflow * jnp.expand_dims(dp_j, axis=-3), axis=(-2,-1))
    inflow_integral = jnp.expand_dims(inflow_integral, axis=-1)
    
    delta_p_j = inflow_integral - outflow_integral
    
    p_j = p_j + step_size * delta_p_j
    p_j = jnp.roll(p_j, shift=1, axis=-1)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j, delta_p_j

@partial(jax.jit, static_argnames=['flow'])
def solve_p_j_heun(p_j_0: jnp.ndarray, 
                   step_sizes: jnp.ndarray, 
                   flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]], 
                   *args: jnp.ndarray,
                   **kwargs: jnp.ndarray):

    def scan_p_j(carry, step_size):
        p_j, t, flow_i = carry
        outflow, inflow = flow_i
        
        p_j_1, delta_p_j_1 = next_p_j(p_j, outflow, inflow, step_size)
        
        t += step_size
        outflow, inflow = flow(t, *args, **kwargs)
        
        _, delta_p_j_2 = next_p_j(p_j_1, outflow, inflow, step_size)
        
        p_j = p_j + 0.5 * step_size * delta_p_j_1
        p_j = jnp.roll(p_j, shift=1, axis=-1) + 0.5 * step_size * delta_p_j_2
        p_j = p_j.at[..., 0].set(0)

        return (p_j, t, (outflow, inflow)), p_j
    
    t0 = jnp.zeros_like(step_sizes[0])
    flow_0 = flow(t0, *args, **kwargs)
    _, p_j = jax.lax.scan(scan_p_j, (p_j_0, t0, flow_0), jnp.expand_dims(step_sizes, axis=-1))
    
    p_j = jnp.concatenate([jnp.expand_dims(p_j_0, axis=0), p_j])
    p_j = jnp.swapaxes(p_j, 0, 1)
    p_j = jnp.swapaxes(p_j, 1, 2)

    return p_j


@partial(jax.jit, static_argnames=['flow'])
def solve(step_sizes: jnp.ndarray, 
          flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]], 
          *args: jnp.ndarray, 
          **kwargs: jnp.ndarray):

    outflow, inflow = flow(0, *args, **kwargs)
    step_sizes = jnp.expand_dims(step_sizes, axis=-1)
    
    p_j_0 = jnp.zeros_like(outflow)
    p_j_0 = p_j_0.at[(..., 0, slice(None))].set(1.0)

    p_j = solve_p_j(p_j_0, step_sizes, flow, *args, **kwargs)
    
    return p_j

@partial(jax.jit, static_argnames=['flow'])
def solve_heun(step_sizes: jnp.ndarray, 
          flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]], 
          *args: jnp.ndarray, 
          **kwargs: jnp.ndarray):

    outflow, inflow = flow(0, *args, **kwargs)
    step_sizes = jnp.expand_dims(step_sizes, axis=-1)
    
    p_j_0 = jnp.zeros_like(outflow)
    p_j_0 = p_j_0.at[(..., 0, slice(None))].set(1.0)

    p_j = solve_p_j_heun(p_j_0, step_sizes, flow, *args, **kwargs)
    
    return p_j
