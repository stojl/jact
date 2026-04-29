# jact/cashflows.py Review

Reviewed file: `jact/cashflows.py`

## Findings

1. Validated declarations retain mutable user mappings.

   `validate_cashflow_components()` validates the mappings inside
   `StateRate`, `TransitionLump`, and `ScheduledEvent`, but then stores the
   original component objects in `CashflowDeclaration`. If the caller keeps a
   reference to the original dict and mutates it after validation, the
   declaration can change after passing validation. Suggested change: freeze
   each payment mapping into a plain `dict` copy or `MappingProxyType`, wrapped
   in a new frozen component object.

2. Zero-dimensional JAX scalar weights are rejected at validation.

   `_validate_view_common()` accepts Python numbers or callables, while solver
   weighting later can handle values through `jnp.asarray()`. Passing
   `weight=jnp.array(0.5)` is therefore rejected even though the runtime path
   could evaluate it. Suggested change: either document Python scalar-only
   weights or accept rank-0 array-like scalar weights explicitly.

3. `Group.members` can be a mutable sequence.

   `Group` is a frozen dataclass, but `members` may be a list. The object cannot
   be reassigned, but the list contents can still be mutated before solve-time
   validation. Suggested change: normalize members to a tuple during view
   validation or in a `__post_init__`.

## Tests To Add

- Mutation-after-validation test for cashflow component payment mappings.
- Explicit test for rank-0 array weights, either accepted or rejected with a
  documented error.

