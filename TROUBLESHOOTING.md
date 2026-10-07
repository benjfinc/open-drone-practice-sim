# Troubleshooting

## Setup fails

The supported interpreter is 64-bit Python 3.12.

On Ubuntu, install Python's virtual-environment support and common OpenGL libraries:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv libgl1 libegl1
python3.12 bootstrap.py
```

If `python3` selects another version, invoke `python3.12` explicitly or use
`PYTHON_BIN=python3.12 ./setup.sh`.

On Windows, install 64-bit Python 3.12 and enable the Python launcher, then run:

```powershell
py -3.12 bootstrap.py
```

If PowerShell blocks local `.ps1` scripts, use the Python commands above; changing the
machine execution policy is not required. If setup was interrupted, rerun the same
bootstrap command. It safely reuses `.venv`.

## The window does not open

The simulator requires OpenGL 3.3 and a working desktop graphics driver.

- On Windows, install the current driver directly from NVIDIA, AMD, or Intel and run
  `py run.py --keyboard` from PowerShell so any error remains visible.
- On Linux, inspect the active renderer with `glxinfo -B`. Install `mesa-utils` if
  that command is unavailable. The tested visible path uses X11/XWayland and GLX.

Please include the full terminal error when reporting a startup failure.

## The wrong Linux GPU is used

On a hybrid NVIDIA laptop, request PRIME offload explicitly:

```bash
DRONE_SIM_PRIME_OFFLOAD=1 ./run.sh
```

The simulator otherwise uses the system's default OpenGL device.

## Fullscreen or resizing looks wrong

Use `F11` or `Alt+Enter` to toggle fullscreen. The renderer follows the actual OpenGL
drawable size and recomputes its HUD layout after a mode change. You can also start in
fullscreen:

```text
python run.py --set display.fullscreen=true
```

If the problem remains, report the operating system, desktop scaling percentage,
display resolution, GPU/driver, and a screenshot.

## A controller is not detected

Confirm the transmitter is in USB joystick/HID mode and inspect raw input.

Linux:

```bash
.venv/bin/python calibrate.py --show
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe calibrate.py --show
```

Remove `--show` to start guided calibration. The RadioMaster Zorro has been tested on
Linux; Windows transmitter handling is expected through SDL but not yet hardware-tested.
Press `K` to use keyboard flight while diagnosing a controller.

Calibration is local to `config/calibration.yaml` and is excluded from Git.

## PowerShell cannot find `py`

Try `python --version`. If it reports Python 3.12, substitute `python` for `py`:

```powershell
python bootstrap.py
python run.py
```

Otherwise reinstall 64-bit Python 3.12 with the Python launcher enabled.

## Logs consume too much disk space

Normal runs record 1 kHz state data under `runs/`. Disable logging with:

```text
python run.py --no-log
```

Run data is ignored by Git and can be removed independently of the simulator.

## Reporting a problem

Include the operating system, Python version, GPU and driver, controller model,
terminal output, exact command, and whether keyboard mode works. Run
`python run.py --help` for launch options.
