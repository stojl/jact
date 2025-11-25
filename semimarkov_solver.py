import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Sequence, Any, Dict, Optional, Union

@jax.jit
def roll_probability_tensor(p: jnp.ndarray):
      p = jnp.roll(p, shift=1, axis=-1)
      p = p.at[..., 0].set(0)
      
      return p

@jax.jit
def update_p(p: jnp.ndarray, delta: jnp.ndarray, step_size: jnp.ndarray):
      p = p + jnp.expand_dims(step_size, axis=-1) * delta
      p = roll_probability_tensor(p)
      return p

@jax.jit 
def update_p_point(p: jnp.ndarray, delta: jnp.ndarray, step_size: jnp.ndarray):
      p = p + step_size * delta
      p = roll_probability_tensor(p)
      return p

@jax.jit
def compute_derivative(p: jnp.ndarray,
                       p_point: jnp.ndarray,
                       mu_plus: jnp.ndarray,
                       mu_minus: jnp.ndarray):
      
      outflow_plus, inflow_plus = mu_plus
      outflow_minus, inflow_minus = mu_minus
      
      outflow_plus = jnp.sum(outflow_plus, axis=-2)
      outflow_minus = jnp.sum(outflow_minus, axis=-2)
      
      dp = jnp.diff(p, axis=-1) # B X J X (D - 1)
      dp_point = jnp.diff(p_point, axis=-1, prepend=0) # B X D
      
      outflow_avg = 0.5 * (outflow_plus[..., :-1] + outflow_minus[..., 1:]) # B X J X (D - 1)
      inflow_avg = 0.5 * (inflow_plus[..., :-1] + inflow_minus[..., 1:]) # B X J X J X (D - 1)
      
      outflow_integral = jnp.cumulative_sum(outflow_avg * dp, axis=-1, include_initial=1) # B X J X D
      inflow_integral = jnp.sum(inflow_avg * jnp.expand_dims(dp, axis=-3), axis=(-2,-1)) # B X J
      inflow_integral = jnp.expand_dims(inflow_integral, axis=-1) # B X J X 1
      
      outflow_point_integral = jnp.cumsum(outflow_plus[..., 0, :] * dp_point, axis=-1) # B X D
      
      inflow_point_integral = inflow_plus[..., 0, :] * jnp.expand_dims(dp_point, axis=-2) # B X J X D
      inflow_point_integral = jnp.sum(inflow_point_integral, axis=-1, keepdims=True) # B X 1
      
      inflow_integral = inflow_integral + inflow_point_integral
      
      delta_p = inflow_integral - outflow_integral
      delta_p_point = -outflow_point_integral
      
      return delta_p, delta_p_point            

@jax.jit
def compute_cashflow(p: jnp.ndarray,
                     p_point: jnp.ndarray,
                     c_plus: jnp.ndarray,
                     c_minus: jnp.ndarray,
                     mu_plus: jnp.ndarray,
                     mu_minus: jnp.ndarray):
      
      dp = jnp.diff(p, axis=-1) # B X J X (D - 1)
      dp_point = jnp.diff(p_point, axis=-1, prepend=0) # B X D
      
      dB_plus, B_jump_plus = c_plus
      dB_minus, B_jump_minus = c_minus
      
      outflow_plus, _ = mu_plus
      outflow_minus, _ = mu_minus
      
      dB_all_plus = dB_plus + jnp.sum(B_jump_plus * outflow_plus, axis=-2)
      dB_all_minus = dB_minus + jnp.sum(B_jump_minus * outflow_minus, axis=-2)
      
      dB_all_avg = 0.5 * (dB_all_plus[..., :-1] + dB_all_minus[..., 1:])
      dB_int = jnp.sum(dB_all_avg * dp + dB_all_plus * dp_point, axis=-1)
      
      return dB_int      

@partial(jax.jit, static_argnames=['flow'])
def solve_p(p_0: jnp.ndarray, 
            p_point_0: jnp.ndarray, 
            grid: jnp.ndarray, 
            flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
            cashflow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
            pertubation: jnp.ndarray,
            *args: jnp.ndarray, 
            **kwargs: jnp.ndarray):
      
      grid_minus = grid - pertubation
      grid_plus = grid + pertubation
      
      step_sizes = jnp.diff(grid, axis=-1)
      step_sizes = jnp.swapaxes(step_sizes, 0, 1)
      step_sizes = jnp.expand_dims(step_sizes, axis=-1)
      
      def heun_scan(carry, step_size):
            p, p_point, t = carry
            
            t_plus = t + pertubation
            
            c_plus = cashflow(t_plus, grid_plus)
            c_minus = cashflow(t_plus, grid_minus)
            
            mu_plus = flow(t_plus, grid_plus, *args, **kwargs)
            mu_minus = flow(t_plus, grid_minus, *args, **kwargs)
            
            # Left end point of cashflow
            dB_left = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            
            delta_p, delta_p_point = compute_derivative(p, p_point, mu_plus, mu_minus)
            
            t += step_size
            
            p_2 = update_p(p, delta_p, step_size)
            p_point_2 = update_p_point(p_point, delta_p_point, step_size)
            
            t_minus = t - pertubation
            
            mu_plus = flow(t_minus, grid_plus, *args, **kwargs)
            mu_minus = flow(t_minus, grid_minus, *args, **kwargs)
            
            delta_p_2, delta_p_point_2 = compute_derivative(p_2, p_point_2, mu_plus, mu_minus)
            
            delta_p_2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
            delta_p_point2 = 0.5 * (delta_p_point_2[..., 1:] + delta_p_point[..., :-1])
            
            delta_p = delta_p.at[..., :-1].set(delta_p_2)
            delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)
            
            p = update_p(p, delta_p, step_size)
            p_point = update_p_point(p_point, delta_p_point, step_size)
            
            c_plus = cashflow(t_minus, grid_plus)
            c_minus = cashflow(t_minus, grid_minus)
            
            # Right end point of cashflow
            dB_right = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            dA = 0.5 * (dB_left + dB_right) * step_size
            
            next_carry = (p, p_point, t)
            history = (p, p_point, dA)
            
            return next_carry, history
      
      t0 = jnp.zeros_like(step_sizes[0])
      _, result = jax.lax.scan(heun_scan, (p_0, p_point_0, t0), step_sizes)
      
      return result

@partial(jax.jit, static_argnames=['flow'])
def solve_p(p_0: jnp.ndarray, 
            p_point_0: jnp.ndarray, 
            grid: jnp.ndarray, 
            flow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
            cashflow: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
            pertubation: jnp.ndarray,
            *args: jnp.ndarray, 
            **kwargs: jnp.ndarray):
      
      grid_minus = grid - pertubation
      grid_plus = grid + pertubation
      
      step_sizes = jnp.diff(grid, axis=-1)
      step_sizes = jnp.swapaxes(step_sizes, 0, 1)
      step_sizes = jnp.expand_dims(step_sizes, axis=-1)
      
      def heun_scan(carry, step_size):
            p, p_point, t = carry
            
            t_plus = t + pertubation
            
            c_plus = cashflow(t_plus, grid_plus)
            c_minus = cashflow(t_plus, grid_minus)
            
            mu_plus = flow(t_plus, grid_plus, *args, **kwargs)
            mu_minus = flow(t_plus, grid_minus, *args, **kwargs)
            
            # Left end point of cashflow
            dB_left = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            
            delta_p, delta_p_point = compute_derivative(p, p_point, mu_plus, mu_minus)
            
            t += step_size
            
            p_2 = update_p(p, delta_p, step_size)
            p_point_2 = update_p_point(p_point, delta_p_point, step_size)
            
            t_minus = t - pertubation
            
            mu_plus = flow(t_minus, grid_plus, *args, **kwargs)
            mu_minus = flow(t_minus, grid_minus, *args, **kwargs)
            
            delta_p_2, delta_p_point_2 = compute_derivative(p_2, p_point_2, mu_plus, mu_minus)
            
            delta_p_2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
            delta_p_point2 = 0.5 * (delta_p_point_2[..., 1:] + delta_p_point[..., :-1])
            
            delta_p = delta_p.at[..., :-1].set(delta_p_2)
            delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)
            
            p = update_p(p, delta_p, step_size)
            p_point = update_p_point(p_point, delta_p_point, step_size)
            
            c_plus = cashflow(t_minus, grid_plus)
            c_minus = cashflow(t_minus, grid_minus)
            
            # Right end point of cashflow
            dB_right = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            dA = 0.5 * (dB_left + dB_right) * step_size
            
            next_carry = (p, p_point, t)
            history = (p, p_point, dA)
            
            return next_carry, history
      
      t0 = jnp.zeros_like(step_sizes[0])
      _, result = jax.lax.scan(heun_scan, (p_0, p_point_0, t0), step_sizes)
      
      return result

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

@partial(jax.jit, static_argnames=['prob_callback', 'cashflow_callback'])
def construct_callback(prob_callback: Optional[Callable[..., jnp.ndarray]], 
                       cashflow_callback: Optional[Callable[..., jnp.ndarray]]):
      
      if prob_callback is None and cashflow_callback is None:
            raise ValueError("At least one of 'prob_callback' or 'cashflow_callback' must be provided.")
      
      @jax.jit
      def callback(p, p_point, dB_left, dB_right, step_size, t_left, t_right):
            if prob_callback is None:
                  return cashflow_callback(p, p_point, dB_left, dB_right, step_size, t_left, t_right)
            if cashflow_callback is None:
                  return prob_callback(p, p_point)
            else:
                  cf = cashflow_callback(p, p_point, dB_left, dB_right, step_size, t_left, t_right)
                  pr = prob_callback(p, p_point)
                  return (pr, cf)
            
      return callback
      

@partial(jax.jit, static_argnames=['intensity', 'cashflow', 'prob_callback', 'cashflow_callback'])
def heun_scheme_solver(p_0: jnp.ndarray, 
                       p_point_0: jnp.ndarray, 
                       grid: jnp.ndarray, 
                       intensity: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
                       intensity_kwargs: Optional[Dict[str, jnp.ndarray]],
                       cashflow: Optional[Callable[..., tuple[jnp.ndarray, jnp.ndarray]]],
                       cashflow_kwargs: Optional[Dict[str, jnp.ndarray]],
                       prob_callback: Callable[..., jnp.ndarray],
                       cashflow_callback: Callable[..., jnp.ndarray],
                       pertubation: jnp.ndarray):
      
      grid_minus = grid - pertubation
      grid_plus = grid + pertubation
      
      step_sizes = step_sizes_from_grid(grid)
      
      callback_fn = construct_callback(prob_callback, cashflow_callback)
      
      def heun_scan(carry, step_size):
            p, p_point, t = carry
            
            t_left = t + pertubation
            
            mu_plus = intensity(t_left, grid_plus, **intensity_kwargs)
            mu_minus = intensity(t_left, grid_minus, **intensity_kwargs)
            
            if cashflow is not None:
                  c_plus = cashflow(t_left, grid_plus, **cashflow_kwargs)
                  c_minus = cashflow(t_left, grid_minus, **cashflow_kwargs)
            
                  dB_left = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            
            delta_p, delta_p_point = compute_derivative(p, p_point, mu_plus, mu_minus)
            
            t_right += step_size
            
            p_2 = update_p(p, delta_p, step_size)
            p_point_2 = update_p_point(p_point, delta_p_point, step_size)
            
            t_minus = t - pertubation
            
            mu_plus = intensity(t_right, grid_plus, **intensity_kwargs)
            mu_minus = intensity(t_right, grid_minus, **intensity_kwargs)
            
            delta_p_2, delta_p_point_2 = compute_derivative(p_2, p_point_2, mu_plus, mu_minus)
            
            delta_p_2 = 0.5 * (delta_p_2[..., 1:] + delta_p[..., :-1])
            delta_p_point2 = 0.5 * (delta_p_point_2[..., 1:] + delta_p_point[..., :-1])
            
            delta_p = delta_p.at[..., :-1].set(delta_p_2)
            delta_p_point = delta_p_point.at[..., :-1].set(delta_p_point2)
            
            p = update_p(p, delta_p, step_size)
            p_point = update_p_point(p_point, delta_p_point, step_size)
            
            if cashflow is not None:
                  c_plus = cashflow(t_minus, grid_plus, **cashflow_kwargs)
                  c_minus = cashflow(t_minus, grid_minus, **cashflow_kwargs)
            
                  dB_right = compute_cashflow(p, p_point, c_plus, c_minus, mu_plus, mu_minus)
            
            next_carry = (p, p_point, t)
            history = callback_fn(p, p_point, dB_left, dB_right, step_size, t_left, t_right)
            
            return next_carry, history
      
      t0 = jnp.zeros_like(step_sizes[0])
      
      _, result = jax.lax.scan(heun_scan, (p_0, p_point_0, t0), step_sizes)

@partial(jax.jit, static_argnames=['intensity', 'cashflow'])
def semimarkov_solver(grid: jnp.ndarray, 
                      intensity: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
                      intensity_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
                      cashflow: Optional[Callable[..., tuple[jnp.ndarray, jnp.ndarray]]] = None,
                      cashflow_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
                      pertubation: jnp.ndarray = 1e-12):
      
      if intensity_kwargs is None:
            intensity_kwargs = {}
      
      if cashflow_kwargs is None:
            cashflow_kwargs = {}
      
      outflow, inflow = intensity(0, grid, **intensity_kwargs)
      
      p_point_0 = jnp.ones_like(outflow[..., 0, :])
      p_0 = jnp.zeros_like(outflow)
      
      val = heun_scheme_solver(
            p_0,
            p_point_0,
            grid,
            intensity,
            intensity_kwargs,
            cashflow,
            cashflow_kwargs,
            pertubation
      )
      
      p, p_point = solve_p(p_0, p_point_0, grid, intensity, pertubation, intensity_kwargs)
      
      p_point = jnp.concatenate([jnp.expand_dims(p_point_0, axis=0), p_point])
      p_point = jnp.swapaxes(p_point, 0, 1)
      
      p = jnp.concatenate([jnp.expand_dims(p_0, axis=0), p])
      p = jnp.swapaxes(p, 0, 1)
      p = jnp.swapaxes(p, 1, 2)
      
      return p, p_point
