# Contributing

Contributions that improve portability, controller support, track authoring,
simulation correctness, documentation, or tests are welcome.

## Development setup

Clone the repository and run the same cross-platform bootstrap used by end users:

```text
git clone https://github.com/benjfinc/open-drone-practice-sim.git
cd open-drone-practice-sim
python bootstrap.py
```

Use `python3.12 bootstrap.py` on Linux or `py -3.12 bootstrap.py` on Windows when
`python` does not already select Python 3.12. Create a focused branch and keep
generated runs and calibration files out of Git.

## Before opening a pull request

Activate `.venv`, then run:

```text
python -m pytest -q
python -m fpvsim --list-tracks
```

On Linux, also validate the wrappers with `bash -n run.sh setup.sh`. Rendering changes
must be checked in a visible window at multiple sizes and in fullscreen. Controller
changes should state the hardware model, operating system, and whether guided
calibration was exercised.

Keep each change focused, add tests for behavior changes, and update user-facing docs
when controls, setup, configuration, or compatibility changes.

## Assets, privacy, and licensing

New tracks and assets must be original or clearly redistributable. Do not submit
private competition maps, branded textures without permission, recorded user runs,
secrets, machine-specific paths, or personal calibration files.

By contributing, you agree that your contribution can be distributed under this
project's [GPL-3.0-or-later license](LICENSE). Preserve relevant upstream attribution
and update [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) when adding third-party
code or assets.

## Bug reports

Include the operating system, Python version, GPU and driver, controller model,
terminal output, exact command, and a minimal reproduction. Screenshots are especially
helpful for rendering and display-mode problems.
