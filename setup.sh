#!/usr/bin/env bash
# Linux convenience wrapper for the cross-platform Python bootstrap.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.12 and its venv module first." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "Open Drone Racing Sim currently requires Python 3.12." >&2
  echo "Found: $($PYTHON_BIN --version 2>&1)" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$HERE/bootstrap.py" "$@"
