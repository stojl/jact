"""Type-check the public consumer against an installed wheel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel to install and check")
    parser.add_argument(
        "--python-version",
        required=True,
        choices=("3.10", "3.12"),
        help="Pyright semantics and interpreter version required for this check",
    )
    return parser


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    args = _parser().parse_args()
    wheel = args.wheel.resolve(strict=True)
    requested_version = tuple(int(part) for part in args.python_version.split("."))
    actual_version = sys.version_info[:2]
    if actual_version != requested_version:
        raise SystemExit(
            "installed-wheel check requires Python "
            f"{args.python_version}, not {actual_version[0]}.{actual_version[1]}"
        )

    repository = Path(__file__).resolve().parents[1]
    consumer = repository / "tests" / "typing_consumer.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    with tempfile.TemporaryDirectory(prefix="jact-wheel-typing-") as directory:
        root = Path(directory)
        environment_dir = root / "venv"
        venv.EnvBuilder(with_pip=False).create(environment_dir)
        if os.name == "nt":
            python = environment_dir / "Scripts" / "python.exe"
        else:
            python = environment_dir / "bin" / "python"

        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(python),
                "install",
                "--disable-pip-version-check",
                str(wheel),
                "pyright>=1.1.0",
                "typing_extensions>=4.0",
            ],
            cwd=root,
            environment=environment,
        )

        shutil.copy2(consumer, root / consumer.name)
        config = {
            "include": [consumer.name],
            "venvPath": ".",
            "venv": environment_dir.name,
            "pythonVersion": args.python_version,
            "typeCheckingMode": "standard",
        }
        (root / "pyrightconfig.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

        verification = (
            "from pathlib import Path; import jact, sys; "
            "package = Path(jact.__file__).resolve(); "
            "environment = Path(sys.prefix).resolve(); "
            "assert environment in package.parents, "
            "f'{package} is not installed in {environment}'; "
            "print(f'checking installed package: {package}')"
        )
        _run(
            [str(python), "-c", verification],
            cwd=root,
            environment=environment,
        )
        _run(
            [str(python), "-m", "pyright", "--project", "pyrightconfig.json"],
            cwd=root,
            environment=environment,
        )


if __name__ == "__main__":
    main()
