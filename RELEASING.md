# Releasing `jact`

This project publishes to PyPI from Git tags using GitHub Trusted Publisher.

## One-time setup

1. Create the `jact` project on PyPI if it does not exist yet.
2. In PyPI, add a Trusted Publisher for this repository.
3. Configure the publisher to trust the workflow
   `.github/workflows/publish.yml`.

The workflow uses OIDC and does not require a long-lived PyPI API token.

## Release checklist

1. Make sure the working tree is clean.
2. Update `version` in `pyproject.toml`.
3. Review the release notes you want to publish with the tag.
4. Run the local validation commands:

```bash
rm -rf build dist src/*.egg-info
pip install -e '.[dev]'
python -m build --no-isolation
python -m twine check dist/*
pytest tests/test_solver.py tests/test_initial_distribution_integration.py -q
```

5. Commit the version bump and related release notes.
6. Create an annotated tag named `vX.Y.Z` that matches `pyproject.toml`.
7. Push the commit and tag to GitHub:

```bash
git push origin main
git push origin vX.Y.Z
```

8. Confirm that the `publish` workflow succeeds on GitHub.
9. Verify the published package:

```bash
python -m venv /tmp/jact-release-check
/tmp/jact-release-check/bin/pip install jact
/tmp/jact-release-check/bin/python -c "import jact; print(jact.__version__)"
```

## Notes

- The release workflow rebuilds the sdist and wheel on GitHub before upload.
- The release workflow runs the same core solver/integration test slice used in
  the checklist above.
- `twine check` is part of the release gate to catch metadata and README
  rendering issues before publish.
- If you need a dry run before the first real release, publish the same artifacts
  to TestPyPI from a temporary workflow change or local manual upload.
