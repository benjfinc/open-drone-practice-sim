#!/usr/bin/env python3
"""Create the local environment and install this source checkout."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="install without running pytest")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Python 3.12 is required; found {sys.version.split()[0]}")
    python = venv_python()
    if not python.exists():
        print(f"Creating {VENV} with {sys.version.split()[0]}...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-e", ".[dev]"])
    if not args.skip_tests:
        run([python, "-m", "pytest", "-q"])
    print("\nSetup complete. Start with:")
    print("  python run.py")


if __name__ == "__main__":
    main()
