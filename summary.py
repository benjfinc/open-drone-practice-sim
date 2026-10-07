#!/usr/bin/env python3
"""Summarise a run (default: the newest one in runs/).

    .venv/bin/python summary.py [runs/<timestamp>] [--export-camera cams.npz --rate 30]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fpvsim.loader import camera_poses, load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--export-camera", default=None, help="write camera poses npz for re-rendering")
    ap.add_argument("--rate", type=float, default=30.0)
    a = ap.parse_args()
    if a.run:
        path = Path(a.run)
    else:
        runs = sorted((HERE / "runs").glob("2*"))
        if not runs:
            sys.exit("no runs")
        path = runs[-1]
    r = load_run(path)
    print(f"run {path}")
    print(f"  git {r.meta.get('git', {}).get('commit')}  renderer {r.meta.get('gl_renderer')}")
    n = len(r.t)
    if n == 0:
        print("  no ticks logged")
        return
    dt = np.diff(r.t)
    armed = r["armed"] > 0.5
    v = np.linalg.norm(r.get("vx", "vy", "vz"), axis=1)
    w = np.degrees(np.abs(r.get("wx", "wy", "wz")))
    print(f"  ticks {n}  sim span {r.t[-1] - r.t[0]:.2f} s  logged {n * np.median(dt):.2f} s"
          f"  median dt {np.median(dt) * 1e3:.3f} ms  armed {armed.mean() * 100:.0f}%")
    print(f"  max speed {v.max():.1f} m/s  max |rate| roll/pitch/yaw {w[:, 0].max():.0f}/{w[:, 1].max():.0f}/"
          f"{w[:, 2].max():.0f} deg/s  max alt {r['pz'].max():.1f} m")
    if r.frames is not None:
        ft = r.frames[:, 0]
        wall = r.frames[:, -1]
        fps = (len(wall) - 1) / max(wall[-1] - wall[0], 1e-9)
        print(f"  frames {len(ft)}  mean fps {fps:.0f}")
    c = Counter(e["kind"] for e in r.events)
    print("  events " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if r.laps:
        laps = r.laps["laps"]
        print(f"  laps {len(laps)} (valid {sum(l['valid'] for l in laps)}), aborted {len(r.laps['aborted'])}, "
              f"in-aperture crossings {len(r.laps['crossings'])}")
        for l in laps:
            print(f"    lap {l['lap']}: {l['time_s']:.3f} s  {'VALID' if l['valid'] else 'missed ' + ','.join(l['missed'])}")
    if a.export_camera:
        cp = camera_poses(r, a.rate)
        np.savez(a.export_camera, **cp)
        print(f"  wrote {len(cp['t'])} camera poses at {a.rate} Hz -> {a.export_camera}")


if __name__ == "__main__":
    main()
