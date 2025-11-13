import jax
import jax.numpy as jnp
from typing import Callable, Any, Iterable, Union, List, Dict

def construct_jit_sum_function(
    funcs: Iterable[Callable[..., Any]]
) -> Callable[..., Any]:
    """
    Constructs and JIT-compiles a new function that computes the sum
    of the results of an arbitrary number of input functions.

    Args:
        funcs: An iterable (list or tuple) of functions with the 
               same signature that return a JAX-compatible scalar or array.

    Returns:
        A JIT-compiled function that takes the same arguments as the input 
        functions and returns their sum.
    """

    @jax.jit
    def sum_of_functions(*args, **kwargs):
        return sum(f(*args, **kwargs) for f in funcs)
    
    return sum_of_functions

def construct_jit_stack_function(
    funcs: Iterable[Callable[..., Any]],
    axis: int = 0
) -> Callable[..., Any]:
    """
    Constructs and JIT-compiles a new function that computes the results
    of an arbitrary number of input functions and concatenates them.

    Args:
        funcs: An iterable of functions with the same signature.
        axis: The axis along which the results should be concatenated.
              (e.g., 0 for stacking rows, 1 for stacking columns).

    Returns:
        A JIT-compiled function that takes the same arguments as the 
        input functions and returns their concatenated results.
    """
    
    @jax.jit
    def concat_of_functions(*args, **kwargs):
        results = [f(*args, **kwargs) for f in funcs]

        return jnp.stack(results, axis=axis)

    return concat_of_functions

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

def _get_zero_placeholder(func: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    """Returns a new function that returns zeros of the same shape as func."""
    def zero_placeholder_func(*args, **kwargs):
        return func(*args, **kwargs) * 0.0

    return zero_placeholder_func


def construct_jit_functions_from_intensity_matrix(
    intensity_matrix
):
    
    n = len(intensity_matrix)
    if not all(len(row) == n for row in intensity_matrix):
        raise ValueError("Intensity matrix must be square (n x n).")
    
    all_functions = (f for row in intensity_matrix for f in row)
    global_reference_func = next((f for f in all_functions if f is not None), None)
    
    if global_reference_func is None:
        raise ValueError("The intensity matrix contains only None entries. Cannot construct any function or determine output shape.")
    
    zero_func = _get_zero_placeholder(global_reference_func)
    
    row_sum_functions = []
    for row in intensity_matrix:
        filled_row = [
            f if f is not None else zero_func
            for f in row
        ]
        
        row_sum_func = construct_jit_sum_function(filled_row)
        row_sum_functions.append(row_sum_func)
        
    # Transpose intensity matrix
    columns_of_functions = list(zip(*intensity_matrix))
    
    column_concat_functions = []
    for column in columns_of_functions:
        
        filled_column = [
            f if f is not None else zero_func
            for f in column
        ]
        
        column_concat_func = construct_jit_stack_function(
            filled_column, 
            axis=0
        )
        column_concat_functions.append(column_concat_func)
    
    
    @jax.jit
    def inflow_function(*args, **kwargs) -> jnp.ndarray:
        col_results = [f(*args, **kwargs) for f in column_concat_functions]

        return jnp.stack(col_results, axis=0)
    
    @jax.jit
    def outflow_function(*args, **kwargs) -> jnp.ndarray:
        row_results = [f(*args, **kwargs) for f in row_sum_functions]
        
        return jnp.stack(row_results, axis=0)
    
    return outflow_function, inflow_function
    
    
        
        
    
        
    