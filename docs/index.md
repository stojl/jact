# jact documentation

This directory contains the public documentation for `jact`.

## Start here

- [README quickstart](../README.md#quickstart) pairs transitions with intensities in `jact.Model` and solves for state probabilities.
- [Example notebook](example_notebook.ipynb) walks through a solve and adds cashflows and initial distributions.
- [API specification](api_spec.md) defines the public API.
- [Solver reading guide](solver_guide.md) derives the numerical update and traces the main performance optimizations.
- [Derived fields notebook](derived_fields_notebook.ipynb): share computed covariates across intensities and cashflows, then measure reuse and runtime.
- [Pytree domain modelling notebook](pytree_domain_modeling_notebook.ipynb): use registered domain objects and structured derived fields in a solve.
- [Autodiff notebook](autodiff_notebook.ipynb): differentiate a terminal present value with respect to an intensity parameter.
- [Fitted neural-network notebook](fitted_nn_notebook.ipynb) compares fitted and reference intensities on a shared `StateSpace`.

## Maintainer-only material

Internal design notes, implementation notes, reflections, code reviews, and
historical prototype files live outside `docs/`:

- `notes/`: design and implementation notes, reflections, and review material
- `archive/`: historical prototype code retained for reference
- `tools/`: one-off diagnostics and research utilities
