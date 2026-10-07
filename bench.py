#!/usr/bin/env python3
"""Measure physics and rendering performance and save example screenshots.

    python bench.py                            # offscreen rendering + physics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

from fpvsim import config as C  # noqa: E402
from fpvsim import dynamics as D  # noqa: E402
from fpvsim import gl_platform  # noqa: E402
from fpvsim.inputs import Sticks  # noqa: E402
from fpvsim.logger import RunLogger  # noqa: E402
from fpvsim.sim import Sim  # noqa: E402


def bench_physics(cfg, seconds=3.0):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sim = Sim(cfg, RunLogger(Path(td), cfg["logging"]["chunk_ticks"]))
        st = Sticks()
        st.throttle = 0.0
        sim.set_armed(True, 0.0)
        st.throttle = 0.35
        n_ticks, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            st.roll = 0.3 * math.sin(n_ticks * 1e-3)
            sim.advance(8, st)                     # 8 ticks = one frame at 125 fps
            n_ticks += 8
        el = time.perf_counter() - t0
        sim.close()
    return {"ticks_per_s_with_logging_and_python_batching_8": n_ticks / el,
            "realtime_factor_at_1kHz": n_ticks / el / 1000.0}


def bench_render(cfg, out_dir, seconds=4.0):
    import moderngl
    import pygame as pg
    from fpvsim import render
    pg.font.init()
    W, H = int(cfg["display"]["width"]), int(cfg["display"]["height"])
    ctx = gl_platform.create_standalone_context(moderngl)
    samples = int(cfg["display"].get("msaa", 0))
    ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=samples)],
                         depth_attachment=ctx.depth_renderbuffer((W, H), samples=samples))
    res = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4)])
    sim = Sim(cfg, None)
    rend = render.Renderer(ctx, cfg, sim.env, (W, H), pg)
    hfov = render.spec_hfov_deg(cfg["camera"])
    # a few poses: at spawn; hovering 3 m before gate 5 looking at it; high overview
    g5 = sim.elements["5"]
    views = {"fpv_spawn": (sim.s.copy(), "fpv"),
             "fpv_gate5": (None, "fpv"), "chase_gate5": (None, "chase"), "static": (None, "static")}
    s5 = D.new_state((g5.pos[0] - 3 * math.cos(g5.yaw), g5.pos[1] - 3 * math.sin(g5.yaw), 1.35), g5.yaw)
    from PIL import Image
    saved = []
    for name, (s, mode) in views.items():
        s = s5 if s is None else s
        rend.chase_pos = None
        for _ in range(30):
            rend.update_osd({"tl": ["LAP --", "BEST --"], "tr": ["NEON CIRCUIT"], "bl": ["DISARMED"],
                             "top": "NEXT 5", "thr": 0.3})
            rend.render(ms, s, mode, 20.0, hfov, (g5.pos[0], g5.pos[1], 1.35), osd=True)
        ctx.copy_framebuffer(res, ms)
        img = Image.frombytes("RGBA", (W, H), res.read(components=4)).transpose(Image.FLIP_TOP_BOTTOM)
        p = Path(out_dir) / f"bench_{name}.png"
        img.convert("RGB").save(p)
        saved.append(str(p))
    # timed loop: moving fpv camera, OSD refreshed at osd_hz
    s = s5.copy()
    n, t0, last_osd = 0, time.perf_counter(), 0.0
    while time.perf_counter() - t0 < seconds:
        s[1] = g5.pos[1] - 6 + (n % 600) * 0.01
        now = time.perf_counter()
        if now - last_osd > 1 / cfg["display"]["osd_hz"]:
            last_osd = now
            rend.update_osd({"tl": [f"LAP {now - t0:.2f}"], "tr": [f"{n} frames"], "thr": 0.3})
        rend.render(ms, s, "fpv", 20.0, hfov, (g5.pos[0], g5.pos[1], 1.35), osd=True)
        ctx.copy_framebuffer(res, ms)
        ctx.finish()                                  # force GPU completion so fps is honest
        n += 1
    el = time.perf_counter() - t0
    return {"renderer": ctx.info["GL_RENDERER"], "resolution": [W, H], "msaa": samples,
            "fps_offscreen_incl_finish": n / el, "screenshots": saved, "gate_source": rend.gate_source}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "bench_out"))
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = C.load_config(sets=a.set)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = {"physics": bench_physics(cfg)}
    print(json.dumps(out, indent=1))
    out["render"] = bench_render(cfg, a.out)
    print(json.dumps(out["render"], indent=1))
    Path(a.out, "bench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
