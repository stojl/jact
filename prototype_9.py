import jax
import jax.numpy as jnp
from functools import partial, reduce
from typing import Callable, Sequence, Any, Dict, Optional, Union
from probability_callbacks import ProbabilityCallbacks
from cashflow_callbacks import CashflowCallbacks
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


def _compute_core(p, p_point, mu_plus_matrix, mu_minus_matrix):
    J, D_minus_1 = p.shape
    
    outflow_plus_list = [[] for _ in range(J)]
    outflow_avg_list = [[] for _ in range(J)]
    next_inflow_list = []

    for j in range(J):
        inflow_terms_for_j = []
        for i in range(J):
            m_p = mu_plus_matrix[i][j]
            m_m = mu_minus_matrix[i][j]
            
            if m_p is not None and j != i:
                m_p_slice = m_p[:-1]
                m_avg = 0.5 * (m_p_slice + m_m[1:])

                outflow_plus_list[i].append(m_p_slice)
                outflow_avg_list[i].append(m_avg)

                term_p = jnp.sum(m_avg * p[i, :])
                inflow_terms_for_j.append(term_p)
                
                if i == 0:
                    term_p_point = jnp.sum(m_p_slice * p_point)
                    inflow_terms_for_j.append(term_p_point)

        if inflow_terms_for_j:
            next_inflow_list.append(reduce(jax.lax.add, inflow_terms_for_j))
        else:
            next_inflow_list.append(0.0)

    final_outflow_plus = jnp.stack([reduce(jax.lax.add, l) if l else jnp.zeros(D_minus_1) for l in outflow_plus_list])
    final_outflow_avg = jnp.stack([reduce(jax.lax.add, l) if l else jnp.zeros(D_minus_1) for l in outflow_avg_list])

    next_inflow = jnp.array(next_inflow_list)
    
    delta_p = -p * final_outflow_avg
    delta_p_point = -final_outflow_plus[0, :] * p_point 

    return next_inflow, delta_p, delta_p_point

@jax.jit
def compute_derivative(p, p_point, mu_plus_matrix, mu_minus_matrix):
    mu_axes = tuple(tuple(0 if entry is not None else None for entry in row) 
                    for row in mu_plus_matrix)
    vmap_func = jax.vmap(_compute_core, in_axes=(0, 0, mu_axes, mu_axes))
    
    return vmap_func(p, p_point, mu_plus_matrix, mu_minus_matrix)

def compute_jump_payments(mu, payments):
    result_matrix = []

    for m_row, p_row in zip(mu, payments):
        if p_row is None or m_row is None:
            result_matrix.append(None)
            continue
            
        current_row_res = []
        
        for m_val, b_val in zip(m_row, p_row):
            if m_val is None or b_val is None:
                current_row_res.append(None)
            else:
                prod = jax.tree_util.tree_map(lambda x: m_val * x, b_val)
                current_row_res.append(prod)
        
        result_matrix.append(tuple(current_row_res))
        
    return tuple(result_matrix)

def trapezoid_increment(left, right):
    return 0.5 * (left[..., :-1] + right[..., 1:])

def compute_cashflow(
    p, p_point,
    mu_plus, mu_minus,
    cf_plus, cf_minus,
    cashflow_callback):
    
    _, J, _ = p.shape

    cf_plus['jump'] = compute_jump_payments(mu_plus, cf_plus['jump'])
    cf_minus['jump'] = compute_jump_payments(mu_minus, cf_minus['jump'])
    
    cf_plus = cashflow_callback(**cf_plus)
    cf_minus = cashflow_callback(**cf_minus)
    
    cf_avg = jax.tree_util.tree_map(lambda a, b: trapezoid_increment(a, b), cf_plus, cf_minus)
    
    for j in range(J):
        dp = p[:, j, ...]
        cf_avg_j = cf_avg[j]
        
        cf_avg_j = 
        
        


def compute_cashflow_optimized(
    p, p_point, 
    mu_p_matrix, mu_m_matrix,      # Intensities (Tuples of Tuples)
    dB_p_tuple, dB_m_tuple,        # Continuous cashflow (Tuples)
    Bj_p_matrix, Bj_m_matrix       # Jump cashflow (Tuples of Tuples)
):
    """
    p: (B, J, D-1)
    p_point: (B, D-1)
    All matrices/tuples are pre-evaluated or static function handles.
    """
    B, J, D_minus_1 = p.shape
    
    # This will store the combined cashflow rate per state i: (B, J, D-1)
    state_cashflow_rates = []

    for i in range(J):
        db_p = dB_p_tuple[i]
        db_m = dB_m_tuple[i]
        
        # Slicing to D-1
        db_p_slice = db_p[..., :-1]
        db_avg = 0.5 * (db_p_slice + db_m[..., 1:])
        
        jump_contributions = []
        for j in range(J):
            mu_p = mu_p_matrix[i][j]
            mu_m = mu_m_matrix[i][j]
            bj_p = Bj_p_matrix[i][j]
            bj_m = Bj_m_matrix[i][j]
            
            if mu_p is not None and bj_p is not None:
                rate_p = bj_p[..., :-1] * mu_p[..., :-1]
                rate_m = bj_m[..., 1:] * mu_m[..., 1:]
                jump_contributions.append(0.5 * (rate_p + rate_m))

        if jump_contributions:
            total_rate_i = reduce(jax.lax.add, [db_avg] + jump_contributions)
        else:
            total_rate_i = db_avg
            
        state_cashflow_rates.append(total_rate_i)

    total_cashflow = jnp.zeros((B,))
    for i in range(J):
        term = jnp.sum(state_cashflow_rates[i] * p[:, i, :], axis=-1)
        total_cashflow = total_cashflow + term

    db_p0 = dB_p_tuple[0][..., :-1]
    jump_p0_list = []
    for j in range(J):
        if Bj_p_matrix[0][j] is not None and mu_p_matrix[0][j] is not None:
            jump_p0_list.append(Bj_p_matrix[0][j][..., :-1] * mu_p_matrix[0][j][..., :-1])
    
    if jump_p0_list:
        db_all_p0 = reduce(lax.add, [db_p0] + jump_p0_list)
    else:
        db_all_p0 = db_p0
        
    point_term = jnp.sum(db_all_p0 * p_point, axis=-1)
    
    return total_cashflow + point_term

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
    cashflow: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    cashflow_kwargs: Dict[str, jnp.ndarray],
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
    static_argnames=['units', 'discretization_unit', 'intensity', 'prob_callback', 'cashflow_callback', 'transpose_result'],
)
def semimarkov_solver(
    units: int,
    discretization_unit: int,
    intensity: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]],
    intensity_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    cashflow: Sequence[Sequence[Optional[Callable[..., jnp.ndarray]]]] = None,
    prob_callback: Union[None, str, Callable[..., Any]] = 'default',
    cashflow_callback: Union[None, str, Callable[..., Any]] = 'default',
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
        
    if not callable(cashflow_callback):
        cashflow_callback = CashflowCallbacks.from_str(cashflow_callback)
        
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
