import jax
import jax.numpy as jnp
from functools import reduce
from typing import Union

def trapezoid_increment(left, right):
    return 0.5 * (left + right)

def add_with_nones(x, y):
        if x is None: return y
        if y is None: return x
        return x + y
    
def reduce_cashflow(x):    
    return reduce(lambda a, b: jax.tree_util.tree_map(add_with_nones, a, b), x)

def reduce_state(x):
    row_sums = []
    for row in x:
        leaves = jax.tree_util.tree_leaves(row)
        row_sum = sum(leaves) if leaves else None            
        row_sums.append(row_sum)
        
    return tuple(row_sums)
    

class CashflowCallbacks:
    
    @staticmethod
    @jax.jit
    def none(cont, jump):
        return None
    
    @staticmethod
    @jax.jit
    def default(cont, jump):
        return {'cont': cont, 'jump': jump}
    
    @staticmethod
    @jax.jit
    def reduce_jump(cont, jump):
        jump = reduce_state(jump)
        return CashflowCallbacks.default(cont, jump)
    
    @staticmethod
    @jax.jit
    def reduce_cont(cont, jump):
        cont = reduce_state(cont)
        return CashflowCallbacks.default(cont, jump)
    
    @staticmethod
    @jax.jit
    def reduce(cont, jump):
        cf = CashflowCallbacks.reduce_jump(cont, jump)
        return CashflowCallbacks.reduce_cont(**cf)
    
    @staticmethod
    @jax.jit
    def collapse(cont, jump):
        cf = CashflowCallbacks.reduce(cont, jump)
        return reduce_cashflow(cf)
    
    @staticmethod
    def from_str(str: Union[str, None]):
        match str:
            case 'none':
                return CashflowCallbacks.none
            case None:
                return CashflowCallbacks.none
            case 'default':
                return CashflowCallbacks.default
            case 'reduce_cont':
                return CashflowCallbacks.reduce_cont
            case 'reduce_jump':
                return CashflowCallbacks.reduce_jump
            case 'reduce':
                return CashflowCallbacks.reduce
            case 'collapse':
                return CashflowCallbacks.collapse
            case _:
                return CashflowCallbacks.none