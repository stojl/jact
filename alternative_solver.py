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
def pad_last_axis(x: jnp.ndarray) -> jnp.ndarray:
    num_axes = len(x.shape)
    padding = [(0,0)] * (num_axes - 1) + [(1, 0)]
    return jnp.pad(x, padding)

@jax.jit
def slice_second_to_last(x: jnp.ndarray) -> jnp.ndarray:
    num_axes = len(x.shape)
    slices = [slice(None)] * num_axes
    slices[-2] = 0
    return x[tuple(slices)]

@jax.jit
def update_p_j_static_first(p_j, outflow, inflow, step_size):
    outflow = pad_last_axis(outflow)
    inflow = pad_last_axis(inflow)
    inflow =  slice_second_to_last(inflow)
    p_j = p_j + step_size * (inflow - outflow)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j

@jax.jit
def update_p_j_static(p_j, outflow, inflow, step_size):
    dp_j = jnp.diff(p_j, axis=-1)
    
    outflow_integral = jnp.cumulative_sum(outflow * dp_j, axis=-1, include_initial=True)
    inflow_integral = jnp.sum(inflow * dp_j, axis=(-2,-1))
    inflow_integral = jnp.expand_dims(inflow_integral, axis=-1)
    
    p_j = p_j + step_size * (inflow_integral - outflow_integral)
    p_j = jnp.roll(p_j, shift=1, axis=-1)
    p_j = p_j.at[..., 0].set(0)
    
    return p_j

@partial(jax.jit, static_argnames=['flow', 'step_size'])
def update_p_j_scan(init_p_j, p_j_0, duration_grid, flow, step_size):
    
    def update_p_j(p_j, t):    
        outflow, inflow = flow(t, duration_grid)
        p_j = update_p_j_static(p_j, outflow, inflow, step_size)
        return p_j, p_j
    
    p_j_2, p_j = jax.lax.scan(update_p_j, init_p_j, duration_grid)

    #p_j = jnp.swapaxes(p_j, -3, -2)

    p_j_0 = jnp.expand_dims(p_j_0, axis=-2)
    init_p_j = jnp.expand_dims(init_p_j, axis=-2)
    
    return p_j_2#jnp.concatenate([p_j_0, init_p_j, p_j], axis=-2)

@jax.jit
def init_p_j_ones(x):
    num_axes = len(x.shape)
    indicies = [slice(None)] * num_axes
    indicies[-2] = 0

    slice_tuple = tuple(indicies)
    return x.at[slice_tuple].set(1)

@partial(jax.jit, static_argnames=['flow', 'step_size'])
def solve(step_sizes, flow, *args: Any, **kwargs: Any):
    outflow, inflow = flow(0, *args, **kwargs)

    grid_shape = outflow.shape
    p_j_shape = grid_shape[:-1] + (grid_shape[-1] + 1,)

    p_j = jnp.zeros(p_j_shape, dtype=outflow.dtype)
    p_j = init_p_j_ones(p_j)

    p_j_init = update_p_j_static_first(p_j, outflow, inflow, step_size)
    
    return update_p_j_scan(p_j_init, p_j, duration_grid, flow, step_size)
