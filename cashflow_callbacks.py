import jax
import jax.numpy as jnp
from typing import Union

def trapezoid_increment(left, right, step_size):
    return 0.5 * (left + right) * step_size

class CashflowCallbacks:
    
    @staticmethod
    @jax.jit
    def none(p: jnp.ndarray, 
             p_point: jnp.ndarray, 
             dB_left: jnp.ndarray, 
             dB_right: jnp.ndarray, 
             step_size: jnp.ndarray, 
             t_left: jnp.ndarray, 
             t_right:jnp.ndarray):
        
        return None
    
    @staticmethod
    @jax.jit
    def default(p: jnp.ndarray, 
                p_point: jnp.ndarray, 
                dB_left: jnp.ndarray, 
                dB_right: jnp.ndarray, 
                step_size: jnp.ndarray, 
                t_left: jnp.ndarray, 
                t_right:jnp.ndarray):
        
        return trapezoid_increment(dB_left, dB_right, step_size)
    
    @staticmethod
    def from_str(str: Union[str, None]):
        match str:
            case 'none':
                return CashflowCallbacks.none
            case None:
                return CashflowCallbacks.none
            case 'default':
                return CashflowCallbacks.default
            case _:
                return CashflowCallbacks.none