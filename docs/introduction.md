jact is a framework for computing transition probabilities in multi-state models where transition intensities depend on duration in the current state as well as clock time. It is aimed at the common pipeline where intensities are fitted with arbitrary models and then pushed through a numerical solver to obtain probabilities for large cohorts.

The main design choices are:

- `StateSpace` defines topology only: states and allowed transitions.
- `Model` binds JIT-compatible intensity callables to that topology.
- `solve()` automatically reduces work to the reachable subgraph from the declared initial states.
- The solver always integrates hazards with midpoint quadrature along the transported characteristic.

The midpoint-only kernel is deliberate. It keeps the callable interface simple, avoids asking users to declare continuity metadata they often cannot justify for fitted models, and behaves robustly for irregular hazards such as tree-based or other piecewise models. For smooth hazards the midpoint rule remains second-order. If a hazard has jumps strictly inside traversed grid cells, convergence for that callable can degrade; when possible, align split points in time or duration to the solver grid.

If you want a user-facing walkthrough instead of the API spec, start with
[example_notebook.ipynb](example_notebook.ipynb). The historical quadrature comparison that motivated the midpoint-only choice is kept in
[convergence_notebook.ipynb](convergence_notebook.ipynb) as a design-study notebook.

Future work includes pre-computation protocols for factorizable intensities, built-in parametric hazard functions, and cashflow computation via integral transforms over the duration density.
