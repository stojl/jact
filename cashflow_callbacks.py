import jax
import jax.numpy as jnp
from typing import Union

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
    def from_str(str: Union[str, None]):
        match str:
            case 'none':
                CashflowCallbacks.none
            case None:
                CashflowCallbacks.none
            case _:
                CashflowCallbacks.none