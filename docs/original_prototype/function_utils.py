import jax
import jax.numpy as jnp
from typing import Callable, Sequence, Any, Union
from functools import partial

def _get_zero_placeholder(func: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    """Returns a new function that returns zeros of the same shape as func."""
    def zero_placeholder_func(*args, **kwargs):
        return func(*args, **kwargs) * 0.0

    return zero_placeholder_func

def fill_function_matrix(
        function_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]],
        fill_func = Callable[..., jnp.ndarray]
) -> Sequence[Sequence[Callable[..., jnp.ndarray]]]:
    filled_matrix = [
        [
            (func if func is not None else fill_func)
            for func in row 
        ]
        for row in function_matrix
    ]
    
    return filled_matrix

def get_reference_function(function_matrix):
    all_functions = (f for row in function_matrix for f in row)
    global_reference_func = next((f for f in all_functions if f is not None), None)
    return global_reference_func

def prepare_function_matrix(
    function_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]]
):
    n = len(function_matrix)
    if not all(len(row) == n for row in function_matrix):
        raise ValueError("Intensity matrix must be square (n x n).")
    
    global_reference_func = get_reference_function(function_matrix)
    
    if global_reference_func is None:
        raise ValueError("The intensity matrix contains only None entries. Cannot construct any function or determine output shape.")
    
    zero_func = _get_zero_placeholder(global_reference_func)
    function_matrix = fill_function_matrix(function_matrix, zero_func)
    
    return function_matrix

def construct_func_from_payment_matrix(
    payment_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]]
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Returns a new function that computes the continuous cashflow dB and 
    jump cashflow b_jk according to a n x n payment matrix.
    """
    n = len(payment_matrix)
    if not all(len(row) == n for row in payment_matrix):
        raise ValueError("Intensity matrix must be square (n x n).")
    
    all_functions = (f for row in payment_matrix for f in row)
    global_reference_func = next((f for f in all_functions if f is not None), None)
    
    if global_reference_func is None:
        raise ValueError("The intensity matrix contains only None entries. Cannot construct any function or determine output shape.")
    
    zero_func = _get_zero_placeholder(global_reference_func)
    payment_matrix = fill_function_matrix(payment_matrix, zero_func)
    
    dB = [payment_matrix[i][i] for i in range(n)]
    
    for i in range(n):
        payment_matrix[i][i] = zero_func
    
    @jax.jit
    def dB_b_jk(*args: Any, **kwargs: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        evaluated_dB = [
            func(*args, **kwargs) for func in dB
        ]
        
        evaluated_b_jk = [
            [func(*args, **kwargs)for func in row]
            for row in payment_matrix
        ]
        
        evaluated_dB = jnp.stack(evaluated_dB, axis=-2)
        evaluated_b_jk = [jnp.stack(row, axis=-2) for row in evaluated_b_jk]
        evaluated_b_jk = jnp.stack(evaluated_b_jk, axis=-3)
        
        return evaluated_dB, evaluated_b_jk
    
    return dB_b_jk
    
def construct_func_from_intensity_matrix(
        intensity_matrix: Sequence[Sequence[Callable[..., jnp.ndarray]]]
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Returns a new function that computes the outflow and inflow according to an
    n x n intensity matrix.
    """
    intensity_matrix = prepare_function_matrix(intensity_matrix)
    
    @jax.jit
    def outflow_inflow(*args: Any, **kwargs: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        evaluated_matrix = [
            [func(*args, **kwargs)for func in row]
            for row in intensity_matrix
        ]
        
        outflow = [jnp.stack(row, axis=-2) for row in evaluated_matrix]
        outflow = jnp.stack(outflow, axis=-3)

        tranposed_matrix = list(zip(*evaluated_matrix))
        inflow = [jnp.stack(col, axis=-2) for col in tranposed_matrix]
        inflow = jnp.stack(inflow, axis=-3)       

        return outflow, inflow
    
    return outflow_inflow

def construct_jit_truncated_function(
    func: Callable[[jnp.ndarray, Any], jnp.ndarray],
    threshold: Union[int, float, None] = None
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """
    Creates a JIT-compiled function that executes 'expensive_func' 
    conditionally based on the first scalar argument 't'.

    Signature of the returned function: g(t, *args, **kwargs)

    Args:
        func: The function f(t, *args, **kwargs) -> result.
        threshold: The static value for the t < threshold comparison.
    """
    
    if threshold is None:
        return func
    
    def untruncated_func(t_and_operands):
        t, args, kwargs = t_and_operands
        return func(t, *args, **kwargs)

    def truncated_func(t_and_operands):
        t, args, kwargs = t_and_operands
        
        # For JIT-tracing
        t_zero = jnp.zeros_like(t)
        args_zero = [jnp.zeros_like(a) if isinstance(a, jnp.ndarray) else jnp.zeros(()) for a in args]
        kwargs_zero = {k: jnp.zeros_like(v) if isinstance(v, jnp.ndarray) else jnp.zeros(()) for k, v in kwargs.items()}

        placeholder_result = func(t_zero, *args_zero, **kwargs_zero)
        
        return jnp.zeros_like(placeholder_result)

    @jax.jit
    def conditional_op(t, *args, **kwargs):
        condition = t < threshold 
        
        operand = (t, args, kwargs)
        
        return jax.lax.cond(condition, untruncated_func, truncated_func, operand)

    return conditional_op