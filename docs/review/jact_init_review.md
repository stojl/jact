# jact/__init__.py Review

Reviewed file: `jact/__init__.py`

## Findings

1. Internal callback types are not re-exported at top level.

   The `callbacks` module is exported, but `PointMass` and `StateCarry` are not
   available as `jact.PointMass` / `jact.StateCarry`. Tests import them from
   `jact.callbacks`, so this is not a bug. Suggested change: either keep the
   current explicit module access or document it in the public API examples.

## Tests To Add

- If documentation snippets are tested later, include the top-level docstring
  example so this does not drift again.
