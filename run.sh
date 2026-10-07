#!/usr/bin/env bash
# Launch the simulator. Extra arguments pass directly to fly.py.
# Set DRONE_SIM_PRIME_OFFLOAD=1 to request NVIDIA PRIME GPU offload.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  if [[ -t 0 ]]; then
    printf 'First-run setup is required. Install the pinned dependencies now? [Y/n] '
    read -r answer
    case "${answer:-y}" in
      y|Y|yes|YES) "$HERE/setup.sh" ;;
      *) echo "Setup cancelled. Run ./setup.sh when ready." >&2; exit 1 ;;
    esac
  else
    echo "Virtual environment not found. Run $HERE/setup.sh first." >&2
    exit 1
  fi
fi

if [[ "${DRONE_SIM_PRIME_OFFLOAD:-0}" == "1" ]]; then
  export __NV_PRIME_RENDER_OFFLOAD=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
cd "$HERE"
exec "$HERE/.venv/bin/python" -m fpvsim "$@"
