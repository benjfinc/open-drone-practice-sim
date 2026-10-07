#!/usr/bin/env python3
"""Cross-platform source-checkout launcher."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    if not PYTHON.exists():
        raise SystemExit("Environment not found. Run `python bootstrap.py` first.")
    raise SystemExit(subprocess.call([str(PYTHON), "-m", "fpvsim", *sys.argv[1:]], cwd=ROOT))


if __name__ == "__main__":
    main()
