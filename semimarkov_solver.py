import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def update_p_j_static(p_j, outflow, inflow, step_size):
    dp_j = jnp.diff(p_j, axis=-1)
    
    outflow_integral = jnp.cumulative_sum(outflow * dp_j, axis=-1, include_initial=True)
    inflow_integral = jnp.sum(inflow * dp_j, axis=(-2, -1))
    inflow_integral = jnp.expand_dims(inflow_integral, axis=-1)
    
    p_j = p_j + step_size * (inflow_integral - outflow_integral)
    p_j = jnp.roll(p_j, shift=1, axis=-1)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j

@jax.jit
def update_p_j_static_first(p_j, outflow, inflow, step_size):
    outflow_integral = jnp.pad(outflow, ((0,0), (1,0)))
    inflow_integral = jnp.pad(inflow, ((0,0), (0,0), (1,0)))[:,0,:]
    p_j = p_j + step_size * (inflow_integral - outflow_integral)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j

@partial(jax.jit, static_argnames=['outflow', 'inflow', 'step_size'])
def update_p_j_scan(init_p_j, p_j_0, duration_grid, outflow, inflow, step_size):
    
    def update_p_j(p_j, t):
        outflow_vec = outflow(t, duration_grid)
        inflow_vec = inflow(t, duration_grid)
        p_j = update_p_j_static(p_j, outflow_vec, inflow_vec, step_size)
        
        return p_j, p_j
    
    _, p_j = jax.lax.scan(update_p_j, init_p_j, duration_grid)
    
    p_j = jnp.swapaxes(p_j, 0, 1)
    
    init_p_j = jnp.expand_dims(init_p_j, axis=1)
    p_j_0 = jnp.expand_dims(p_j_0, axis=1)
    return jnp.concatenate([p_j_0, init_p_j, p_j], axis=1)

@partial(jax.jit, static_argnames=['outflow', 'inflow', 'step_size', 'n_states'])
def solve(duration_grid, outflow, inflow, step_size, n_states):
    p_j = jnp.zeros((n_states, duration_grid.shape[0] + 1), dtype=jnp.float64)
    ones = jnp.ones((duration_grid.shape[0] + 1), dtype=jnp.float64)
    
    p_j = p_j.at[0,:].set(ones)
    
    outflow_vec = outflow(0, duration_grid)
    inflow_vec = inflow(0, duration_grid)
    
    p_j_init = update_p_j_static_first(p_j, outflow_vec, inflow_vec, step_size)
    return update_p_j_scan(p_j_init, p_j, duration_grid, outflow, inflow, step_size)