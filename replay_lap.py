#!/usr/bin/env python3
"""Render a recorded lap from the 1 kHz flight log.

By default this selects the fastest valid lap in the newest run and plays it
in real time.  Keys: Escape quits, Space pauses, C cycles FPV/chase/static.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fpvsim import config as C  # noqa: E402
from fpvsim import gl_platform  # noqa: E402
from fpvsim import render  # noqa: E402
from fpvsim.track_env import load as load_track  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", help="run directory (default: newest run)")
    ap.add_argument("--lap", type=int, help="lap number (default: fastest valid lap)")
    ap.add_argument("--mode", choices=render.CAM_MODES, default="fpv")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--fps", type=float, default=60.0, help="display frame rate")
    ap.add_argument("--loop", action="store_true", help="repeat until Escape")
    ap.add_argument("--info", action="store_true", help="print selection without opening a window")
    return ap.parse_args()


def newest_run() -> Path:
    runs = sorted((HERE / "runs").glob("2*"))
    if not runs:
        raise SystemExit("No runs found")
    return runs[-1]


def select_lap(run: Path, number: int | None):
    laps = json.loads((run / "laps.json").read_text())["laps"]
    if number is not None:
        found = [lap for lap in laps if lap["lap"] == number]
        if not found:
            raise SystemExit(f"Lap {number} does not exist in {run}")
        return found[0]
    valid = [lap for lap in laps if lap["valid"]]
    if not valid:
        raise SystemExit(f"No valid laps in {run}")
    return min(valid, key=lambda lap: lap["time_s"])


def load_ticks(run: Path):
    """Load finalized arrays, falling back to crash-safe chunks if necessary."""
    try:
        with np.load(run / "ticks.npz") as z:
            t = z["t"]
            data = z["data"]
            columns = [str(c) for c in z["columns"]]
        if len(t) != len(data):
            raise ValueError("time/data length mismatch")
        return t, data, columns, "ticks.npz"
    except (OSError, EOFError, ValueError, KeyError):
        chunks = run / "chunks"
        tick_files = sorted(chunks.glob("ticks_*.npy"))
        time_files = sorted(chunks.glob("t_*.npy"))
        if not tick_files or not time_files:
            raise SystemExit(f"No readable tick data in {run}")
        data = np.concatenate([np.load(p, mmap_mode="r") for p in tick_files])
        t = np.concatenate([np.load(p, mmap_mode="r") for p in time_files])
        columns = json.loads((run / "meta.json").read_text())["tick_columns"]
        return t, data, columns, f"{len(tick_files)} crash-safe chunks"


def main():
    args = parse_args()
    if args.speed <= 0 or args.fps <= 0:
        raise SystemExit("--speed and --fps must be positive")
    run = Path(args.run).resolve() if args.run else newest_run()
    lap = select_lap(run, args.lap)
    t, data, columns, source = load_ticks(run)
    col = {name: i for i, name in enumerate(columns)}
    required = ("px", "py", "pz", "qw", "qx", "qy", "qz", "cam_tilt_deg", "stick_thr")
    missing = [name for name in required if name not in col]
    if missing:
        raise SystemExit(f"Missing tick columns: {', '.join(missing)}")
    lo = int(np.searchsorted(t, lap["t_start"], side="left"))
    hi = int(np.searchsorted(t, lap["t_end"], side="right"))
    if hi <= lo:
        raise SystemExit("Selected lap is outside the recorded tick data")
    print(f"Run: {run}")
    print(f"Fastest valid lap: {lap['lap']}  {lap['time_s']:.3f} s  "
          f"passed={len(lap['passed'])} missed={len(lap['missed'])}")
    print(f"Replay source: {source}; ticks {lo}:{hi}")
    if args.info:
        return

    cfg = yaml.safe_load((run / "config_used.yaml").read_text())
    disp = cfg["display"]
    gl_platform.configure_environment(disp.get("gpu"))
    import moderngl
    import pygame as pg

    pg.init()
    width, height = int(disp["width"]), int(disp["height"])
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
    pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF,
                        vsync=1 if disp.get("vsync") else 0)
    pg.display.set_caption(f"Recorded lap {lap['lap']} — {lap['time_s']:.3f} s")
    ctx = gl_platform.create_window_context(moderngl)
    env = load_track(cfg)
    renderer = render.Renderer(ctx, cfg, env, (width, height), pg)
    hfov = (render.spec_hfov_deg(cfg["camera"])
            if cfg["camera"]["fov_mode"] == "spec" else float(cfg["camera"]["hfov_wide_deg"]))
    modes = render.CAM_MODES
    mode_index = modes.index(args.mode)
    clock = pg.time.Clock()
    playhead = 0.0
    paused = False
    running = True
    last_wall = time.perf_counter()
    try:
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    running = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    paused = not paused
                elif event.type == pg.KEYDOWN and event.key == pg.K_c:
                    mode_index = (mode_index + 1) % len(modes)
            now = time.perf_counter()
            if not paused:
                playhead += (now - last_wall) * args.speed
            last_wall = now
            if playhead > lap["time_s"]:
                if args.loop:
                    playhead %= lap["time_s"]
                    renderer.chase_pos = None
                else:
                    break
            absolute_t = lap["t_start"] + playhead
            idx = int(np.clip(np.searchsorted(t, absolute_t), lo, hi - 1))
            row = data[idx]
            state = np.zeros(10, dtype=float)
            state[0:3] = row[[col["px"], col["py"], col["pz"]]]
            state[6:10] = row[[col["qw"], col["qx"], col["qy"], col["qz"]]]
            tilt = float(row[col["cam_tilt_deg"]])
            throttle = float(row[col["stick_thr"]])
            renderer.update_osd({
                "tl": [f"RECORDED LAP {lap['lap']}", f"TIME {playhead:6.2f} / {lap['time_s']:.2f}"],
                "tr": [f"{args.speed:g}x", f"CAM {modes[mode_index]}"],
                "bl": ["PAUSED" if paused else "PLAYING"],
                "top": "FASTEST VALID LAP", "thr": throttle,
            })
            renderer.render(ctx.screen, state, modes[mode_index], tilt, hfov, osd=True)
            pg.display.flip()
            clock.tick(args.fps)
    finally:
        pg.quit()


if __name__ == "__main__":
    main()
