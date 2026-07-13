# Development Layout

The repository is organized so that public package documentation is separated
from maintainer-only material.

## Local checks

Install the development dependencies with `pip install -e '.[dev]'`, then run
the same checks required by continuous integration:

```bash
pyright
ruff check src tests
pytest
```

## Primary directories

- `src/jact/`: installable package code
- `tests/`: automated test suite
- `docs/`: public documentation intended for users
- `notes/`: internal design notes, implementation notes, reflections, and reviews
- `archive/`: historical reference material not treated as current docs or API
- `tools/`: ad hoc diagnostics and research scripts for maintainers
- `benchmarks/`: local performance checks

## Packaging intent

PyPI artifacts should contain the package from `src/`, its PEP 561 `py.typed`
marker, and core metadata such as `README.md` and `LICENSE`. Internal notes,
archive material, tools, tests, and benchmarks are repository assets and are
excluded from distribution artifacts.
