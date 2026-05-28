"""Utilities for locating and installing the bundled jact agent skill."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Sequence

_SKILL_RELATIVE_PATH = "agents/jact/SKILL.md"


def _skill_resource() -> Traversable:
    return files("jact").joinpath(_SKILL_RELATIVE_PATH)


def skill_path() -> Path:
    """Return the filesystem path to the bundled skill file."""
    resource = _skill_resource()
    if isinstance(resource, Path):
        return resource
    raise RuntimeError(
        "The bundled skill is not available as a stable filesystem path. "
        "Use `jact-agent-skill print` or `jact-agent-skill install` instead."
    )


def skill_text() -> str:
    """Read the bundled skill markdown."""
    return _skill_resource().read_text(encoding="utf-8")


def install_skill(target: str | Path, *, force: bool = False) -> Path:
    """Copy the bundled skill to ``target / "SKILL.md"``.

    Parameters
    ----------
    target : str or Path
        Directory to create or update.
    force : bool, optional
        Replace an existing ``SKILL.md`` when true.

    Returns
    -------
    Path
        Path to the installed skill file.
    """
    target_dir = Path(target).expanduser()
    destination = target_dir / "SKILL.md"

    if destination.exists() and not force:
        raise FileExistsError(
            f"{destination} already exists. Re-run with --force to replace it."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(skill_text(), encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jact-agent-skill",
        description="Locate, print, or install the bundled jact agent skill.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "path",
        help="Print the installed filesystem path to SKILL.md.",
    )
    subparsers.add_parser(
        "print",
        help="Print the SKILL.md contents to stdout.",
    )

    install = subparsers.add_parser(
        "install",
        help="Copy SKILL.md into a target directory.",
    )
    install.add_argument(
        "--target",
        required=True,
        help="Directory where SKILL.md should be written.",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing SKILL.md at the target.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generic jact agent-skill CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "path":
            print(skill_path())
            return 0
        if args.command == "print":
            print(skill_text(), end="")
            return 0
        if args.command == "install":
            destination = install_skill(args.target, force=args.force)
            print(destination)
            return 0
    except Exception as exc:
        print(f"jact-agent-skill: error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 2
