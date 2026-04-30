# Development Layout

The repository is organized so that public package documentation is separated
from maintainer-only material.

## Primary directories

- `src/jact/`: installable package code
- `tests/`: automated test suite
- `docs/`: public documentation intended for users
- `notes/`: internal design notes, implementation notes, reflections, and reviews
- `archive/`: historical reference material not treated as current docs or API
- `tools/`: ad hoc diagnostics and research scripts for maintainers
- `benchmarks/`: local performance checks

## Packaging intent

PyPI artifacts should contain the package from `src/` plus core metadata such
as `README.md` and `LICENSE`. Internal notes, archive material, tools, tests,
and benchmarks are repository assets and are excluded from distribution
artifacts.
