# jact/__init__.py Review

Reviewed file: `jact/__init__.py`

## Status

No code change is needed here.

## Resolved

1. Callback helper types remain submodule-level API by design.

   The public API spec already documents `jact.callbacks.PointMass` and
   `jact.callbacks.StateCarry` as lower-level objects that stay in their
   submodule rather than being re-exported as `jact.PointMass` /
   `jact.StateCarry`.
