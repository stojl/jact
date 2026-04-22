# InitialDistribution: implementation reflections

This note reflects on the `InitialDistribution` object as specified in [docs/api_spec.md](/home/lucas/Documents/jact/docs/api_spec.md). It is intentionally implementation-focused rather than user-facing.

## Core role

`InitialDistribution` is the boundary object between:

- the user-facing ways of declaring where probability mass starts, and
- the solver-facing representation used to seed point masses at `t = 0`.

That means the implementation should optimize for two things at once:

- a small, predictable public API, and
- a rigid internal structure with explicit invariants.

## Main design constraints

The spec implies a few non-negotiable constraints.

### 1. State topology is static

The set of initial states is structural. It is not inferred from runtime masses, and it must stay fixed across tracing/JIT boundaries.

This matters because:

- model reduction depends on the declared initial-state set,
- zero mass in a declared component still allocates a solver slot, and
- `per_individual(initial_states=None)` must conservatively mean "all model states may be initial".

Implementation consequence: the object should store a static declaration of initial states separately from traced arrays.

### 2. Values are dynamic

Masses, durations, and `per_individual` state indices may be traced arrays. Construction therefore cannot rely on host-side inspection of runtime values except where the API explicitly requires eager validation.

Implementation consequence: split validation into:

- construction-time structural and shape checks,
- solve-time state-space membership and bounds checks.

### 3. The object is state-space-agnostic

The base constructors do not depend on a `StateSpace`. That is a deliberate portability choice.

Implementation consequence:

- store state names as opaque strings in the object,
- do not resolve names to indices until `solve()` or reduction entry,
- keep `StateSpace` helpers as thin wrappers that validate early and then delegate.

### 4. Solver seeding is point-mass based

Each declared initial state maps to a point mass at one per-individual duration value. This is true for all three construction styles.

Implementation consequence: the internal normalized form should look like "declared initial components", not like three unrelated modes.

## Recommended code structure

The cleanest structure is to keep one public object and one canonical internal representation.

### Public module layout

`initial_distribution.py` should own:

- the `InitialDistribution` class,
- convenience constructors `at` and `per_individual`,
- small private validation helpers,
- a private canonicalization step used by `solve()`.

`state_space.py` should only contain the ergonomic wrappers:

- `initial_at`
- `initial_per_individual`
- `initial_distribution`

Those wrappers should validate names eagerly and then construct a normal `InitialDistribution`.

### Internal representation

Use a single immutable dataclass-like object with fields that directly encode the structural contract.

Suggested shape:

```python
@dataclass(frozen=True)
class InitialDistribution:
    _kind: Literal["components", "per_individual"]
    _initial_states: tuple[str, ...] | None
    _components: Mapping[str, ComponentSpec] | None
    _state_indices: ArrayLike | None
    _normalise: bool
```

Where:

```python
@dataclass(frozen=True)
class ComponentSpec:
    mass: ArrayLike
    duration: ArrayLike
```

The important point is not the exact names. The important point is that:

- the structural state declaration is explicit,
- array-bearing fields are separate from structural metadata,
- all construction paths can be lowered into the same solver-facing form.

## Canonicalization strategy

The implementation should have one private method that turns any user construction into a canonical list/tuple of declared components.

For example, conceptually:

```python
CanonicalComponent(
    state_name: str,
    mass: scalar_or_batch_array,
    duration: scalar_or_batch_array,
)
```

Then the three public entry points reduce to:

- `at(state, duration)` -> one canonical component with unit mass
- `InitialDistribution(components=...)` -> direct canonical components
- `per_individual(...)` -> one canonical component per declared initial state, with mass produced from the index array

This unification is valuable because the solver contract does not actually care how the object was constructed. It only needs:

- the declared initial-state set,
- mass per declared state,
- duration per declared state.

## `per_individual` deserves special treatment

This constructor is the one most likely to accumulate accidental complexity.

The safest structure is:

1. Store the traced integer array `states` directly.
2. Store the static tuple `initial_states` exactly as declared, or `None`.
3. Defer expansion into per-state masses until canonicalization at `solve()`-entry.

Why defer expansion:

- it preserves the compact, trace-friendly input form,
- it avoids duplicating shape logic in two places,
- it centralizes the "indices -> one-hot state mass" conversion.

At canonicalization time:

- if `initial_states` is not `None`, use it as the declared state list,
- otherwise use the full `model.states`,
- convert `(batch,)` integer indices into per-state masses via one-hot encoding,
- broadcast `duration` once into the same batch shape,
- assign each declared state its own mass vector and duration vector.

That makes `per_individual` just another source of canonical components.

## Validation split

The spec strongly suggests a two-stage validation model.

### Construction-time validation

This should check only what can be checked without a `StateSpace`:

- `components` is non-empty if supplied
- component payloads contain both `mass` and `duration`
- `mass` and `duration` are shape-compatible
- batch dimensions agree across components
- numeric constraints that can be expressed on traced arrays (`mass >= 0`, `duration >= 0`)
- `per_individual.states` has rank 1 or is scalar-compatible only if the API explicitly allows it
- `initial_states`, if provided, is a tuple of unique strings

Construction-time validation should not try to prove state-name membership against any model.

### Solve-time validation

This should check everything that depends on a concrete model:

- every declared state name exists in the model state space
- `per_individual` indices fall in bounds for the active declared state list
- the distribution batch dimension matches covariate batch size

This split matches the spec and keeps the low-level object portable.

## Normalization should be explicit and isolated

`normalise=True` changes mass values but not the structural declaration. That behavior is easy to get wrong if normalization is spread around the code.

Recommendation:

- implement normalization in exactly one private helper,
- run it only on canonical per-state masses,
- keep zero-total rows unchanged,
- avoid mixing normalization logic with validation or model reduction.

Conceptually:

1. canonicalize to per-state masses,
2. optionally normalize row-wise,
3. seed solver point masses.

That ordering is simpler and makes tests clearer.

## Reduction logic should consume only declared initial states

The reduction path should not inspect realized masses to shrink the graph. The spec explicitly rejects that.

So the reduction API should consume:

- keys of `components`,
- the one state from `at`,
- `initial_states` if present,
- otherwise the full model state list.

This keeps the static-topology invariant intact and prevents trace-dependent graph structure.

## Interaction with the solver

The handoff from `InitialDistribution` to the solver should be narrow.

The solver should receive something like:

- `initial_states`: tuple of reduced-state names in structural order
- `initial_masses`: per-state arrays with common batch shape
- `initial_durations`: per-state arrays with common batch shape

Then solver setup can:

- map state names to reduced indices,
- create `StateCarry.point_mass` for declared states,
- leave undeclared reachable states with `point_mass = None`.

This separation is preferable to letting the solver understand all constructor variants directly.

## Testing considerations

The tests should be structured around invariants, not constructors alone.

Important cases:

- `at()` produces one declared state, unit mass, requested duration
- `components` preserves declared keys even when one component mass is identically zero
- `normalise=True` rescales positive-total rows and leaves zero rows untouched
- scalar and `(batch,)` inputs broadcast consistently
- `per_individual(initial_states=None)` implies the full model state list at solve-time
- `per_individual(initial_states=...)` restricts reduction to the declared subset
- typo in component state name fails at `solve()`, not at base constructor time
- `StateSpace` helper catches the same typo eagerly
- out-of-range traced indices fail at `solve()`

The most important regression to guard against is accidental inference of structure from runtime masses.

## Future-proofing

The spec mentions future support for absolutely continuous initial duration densities. The current implementation should leave room for that without distorting v1.

The clean way to prepare is to separate:

- "declared initial support in state space"
- "mass representation within a declared state"

Right now each state uses a point-mass representation `(mass, duration)`. Later, that could grow into a tagged payload, for example point mass vs duration density, without changing the outer structural contract.

That is another reason to avoid baking the current three constructors directly into solver code.

## Recommended implementation summary

The implementation should center on one invariant:

`InitialDistribution` is a static declaration of candidate initial states plus traced per-individual mass/duration data.

From that, the code structure should be:

1. Public constructors build a small immutable object.
2. A single canonicalization step lowers every construction mode to declared per-state components.
3. Normalization happens only in canonical form.
4. `solve()` performs model-dependent validation and reduction from the declared state set.
5. The solver receives only a normalized solver-facing seed representation, not the original constructor mode.

If that separation is kept strict, the object should stay simple, jit-clean, and extensible.
