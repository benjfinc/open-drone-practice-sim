#!/usr/bin/env python3
"""Interactive FPV racing simulator.

    ./run.sh                  # fly (joystick, window, logging)
    run.sh --set camera.fov_mode=wide --set rates.default_preset=pro

Physics + rate loop run at a fixed 1 kHz in batches between rendered frames;
sticks are sampled immediately before each batch and the frame is drawn right
after it, so input-to-photon is one frame plus the display.

Keys: Tab/H help | T tracks | Esc quit | R reset | P rate preset | [ ] camera tilt |
      C camera mode | F FOV | F11 or Alt+Enter fullscreen | F12 screenshot |
      Space arm | K keyboard flying (arrows roll/pitch, Q/E yaw, W/S throttle)
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fpvsim import config as C  # noqa: E402
from fpvsim import gl_platform  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overlay", action="append", default=[], help="extra YAML merged over the config")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. physics.mass_kg=0.9")
    ap.add_argument("--calib", default=None, help="calibration YAML (default config/calibration.yaml)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--hidden", action="store_true", help="hidden window (testing)")
    ap.add_argument("--duration", type=float, default=0.0, help="auto-quit after N wall seconds")
    ap.add_argument("--script", choices=["takeoff"], default=None,
                    help="scripted sticks instead of the joystick (testing only)")
    ap.add_argument("--screenshot-every", type=float, default=0.0, help="save a PNG every N s")
    ap.add_argument("--keyboard", action="store_true", help="start in keyboard-flying mode")
    ap.add_argument("--track", default=None, help="saved track preset id")
    ap.add_argument("--list-tracks", action="store_true", help="list saved track presets and exit")
    return ap.parse_args()


def argv_for_track(args, track_id):
    """Replace any existing --track argument while preserving other launch options."""
    result = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--track":
            skip_next = True
            continue
        if arg.startswith("--track=") or arg == "--list-tracks":
            continue
        result.append(arg)
    return [*result, "--track", track_id]


def scripted_takeoff(t, st):
    """Deterministic test input: arm at 0.5 s, climb, then gentle forward flight."""
    st.roll = st.yaw = 0.0
    st.pitch = 0.0
    st.throttle = 0.0 if t < 1.0 else (0.45 if t < 1.6 else 0.325)
    if t > 2.5:
        st.pitch = 0.08 if t < 3.2 else 0.0
    return t > 0.5


class KeyboardSticks:
    def __init__(self):
        self.thr = 0.0

    def read(self, pg, st, dt):
        k = pg.key.get_pressed()
        st.roll = (1.0 if k[pg.K_RIGHT] else 0.0) - (1.0 if k[pg.K_LEFT] else 0.0)
        st.pitch = (1.0 if k[pg.K_UP] else 0.0) - (1.0 if k[pg.K_DOWN] else 0.0)
        st.yaw = (1.0 if k[pg.K_e] else 0.0) - (1.0 if k[pg.K_q] else 0.0)
        st.roll *= 0.5
        st.pitch *= 0.5
        st.yaw *= 0.5
        self.thr += ((1.0 if k[pg.K_w] else 0.0) - (1.0 if k[pg.K_s] else 0.0)) * 0.6 * dt
        self.thr = min(max(self.thr, 0.0), 1.0)
        st.throttle = self.thr


HELP_COLUMNS = [
    [
        ("FLIGHT — KEYBOARD", [
            ("Up / Down", "Pitch"),
            ("Left / Right", "Roll"),
            ("W / S", "Throttle up / down"),
            ("Q / E", "Yaw"),
            ("Space", "Arm / disarm"),
        ]),
        ("VIEW", [
            ("C", "FPV / chase / static"),
            ("F", "Normal / wide FOV"),
            ("[ / ]", "Camera angle"),
            ("F11", "Fullscreen"),
            ("Alt+Enter", "Fullscreen"),
        ]),
    ],
    [
        ("SESSION", [
            ("R", "Reset flight"),
            ("P", "Cycle rate preset"),
            ("T", "Select saved track"),
            ("K", "Keyboard / controller"),
            ("F12", "Save screenshot"),
            ("Esc", "Quit"),
        ]),
        ("HELP", [
            ("Tab / H", "Open or close controls"),
        ]),
        ("CONTROLLER", [
            ("USB HID", "RadioMaster Zorro tested"),
        ]),
    ],
]


def main():
    a = parse()
    track_presets = C.load_track_presets()
    if a.list_tracks:
        for preset in track_presets:
            print(f"{preset['id']:<16} {preset['name']} — {preset['description']}")
        return
    cfg = C.load_config(a.config, a.overlay, a.set, track=a.track)
    disp = cfg["display"]
    gl_platform.configure_environment(disp.get("gpu"))
    import moderngl
    import numpy as np
    import pygame as pg

    from fpvsim import render, inputs
    from fpvsim.logger import RunLogger
    from fpvsim.sim import Sim

    calib, calib_path = C.load_calibration(a.calib)
    pg.init()
    W, H = int(disp["width"]), int(disp["height"])
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
    if int(disp.get("msaa", 0)) > 0:
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
        pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, int(disp["msaa"]))
    flags = pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE
    if disp.get("fullscreen"):
        flags |= pg.FULLSCREEN
    if a.hidden:
        flags |= pg.HIDDEN
    pg.display.set_mode((W, H), flags, vsync=1 if disp.get("vsync") else 0)
    track_name = cfg["track"].get("name") or cfg["track"].get("preset") or "Custom Track"
    pg.display.set_caption(f"Open Drone Racing Sim — {track_name}")
    ctx = gl_platform.create_window_context(moderngl)
    print(f"[fpv] GL: {ctx.info['GL_RENDERER']}")

    logger = None
    if cfg["logging"]["enabled"] and not a.no_log:
        logger = RunLogger(C.repo_path(cfg["logging"]["dir"]), cfg["logging"]["chunk_ticks"])
    t_c = time.time()
    sim = Sim(cfg, logger)
    print(f"[fpv] sim ready in {time.time() - t_c:.1f}s; checkpoints {sim.env.sequence}")
    joy = inputs.JoystickInput(calib, pg)
    if logger:
        logger.write_meta(cfg, calib, calib_path, C.REPO, extra={
            "gl_renderer": ctx.info["GL_RENDERER"], "joystick": joy.name, "script": a.script})
        print(f"[fpv] logging to {logger.dir}")
    print(f"[fpv] joystick: {joy.name}  calibration: {calib_path}"
          + ("  (UNVERIFIED default mapping -- run calibrate.py)" if not calib.get("verified") else ""))
    rend = render.Renderer(ctx, cfg, sim.env, (W, H), pg)
    print(f"[fpv] gate appearance: {rend.gate_source}")
    print(f"[fpv] track: {track_name} ({cfg['track'].get('preset', 'custom')})")

    st = inputs.Sticks()
    kb = KeyboardSticks()
    kb_mode = a.keyboard or not joy.connected
    kb_armed = False
    last_preset_pos = None
    last_reset_sw = False
    cam_modes = render.CAM_MODES
    cam_idx = cam_modes.index(cfg["camera"]["mode"])
    fov_mode = cfg["camera"]["fov_mode"]
    show_help_on_start = bool(disp.get("show_help_on_start", True))
    startup_track_menu = a.track is None
    track_menu_open = startup_track_menu
    show_help = show_help_on_start and not track_menu_open
    osd_period = 1.0 / float(disp["osd_hz"])
    dt = sim.dt
    frame = 0
    fps = 0.0
    t_start = last = last_osd = time.perf_counter()
    last_shot = t_start
    accum = 0.0
    lag_events = 0
    running = True
    next_track = None
    current_track = cfg["track"].get("preset")
    track_selection = next((i for i, preset in enumerate(track_presets)
                            if preset["id"] == current_track), 0)
    try:
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    k = ev.key
                    if track_menu_open:
                        if k in (pg.K_ESCAPE, pg.K_t):
                            track_menu_open = False
                            if startup_track_menu:
                                show_help = show_help_on_start
                                startup_track_menu = False
                        elif k in (pg.K_UP, pg.K_w):
                            track_selection = (track_selection - 1) % len(track_presets)
                        elif k in (pg.K_DOWN, pg.K_s):
                            track_selection = (track_selection + 1) % len(track_presets)
                        elif k in (pg.K_RETURN, pg.K_KP_ENTER):
                            selected = track_presets[track_selection]["id"]
                            if selected == current_track:
                                track_menu_open = False
                                if startup_track_menu:
                                    show_help = show_help_on_start
                                    startup_track_menu = False
                            else:
                                next_track = selected
                                running = False
                        last_osd = 0.0
                        continue
                    if k == pg.K_ESCAPE:
                        running = False
                    elif k == pg.K_r:
                        sim.reset("key")
                    elif k == pg.K_p:
                        sim.cycle_preset()
                    elif k == pg.K_LEFTBRACKET:
                        sim.set_tilt(sim.tilt_deg - cfg["camera"]["tilt_step_deg"])
                    elif k == pg.K_RIGHTBRACKET:
                        sim.set_tilt(sim.tilt_deg + cfg["camera"]["tilt_step_deg"])
                    elif k == pg.K_c:
                        cam_idx = (cam_idx + 1) % len(cam_modes)
                    elif k == pg.K_f:
                        fov_mode = "wide" if fov_mode == "spec" else "spec"
                    elif k in (pg.K_TAB, pg.K_h):
                        show_help = not show_help
                        last_osd = 0.0
                    elif k == pg.K_t:
                        track_menu_open = True
                        startup_track_menu = False
                        show_help = False
                        kb_armed = False
                        kb.thr = 0.0
                        sim.set_armed(False, 0.0)
                        last_osd = 0.0
                    elif k == pg.K_F11 or (k in (pg.K_RETURN, pg.K_KP_ENTER)
                                           and ev.mod & pg.KMOD_ALT):
                        try:
                            pg.display.toggle_fullscreen()
                            disp["fullscreen"] = not bool(disp.get("fullscreen"))
                            sim.msg("FULLSCREEN" if disp["fullscreen"] else "WINDOWED")
                        except pg.error as exc:
                            sim.msg(f"FULLSCREEN UNAVAILABLE: {exc}")
                    elif k == pg.K_k:
                        kb_mode = not kb_mode
                        sim.msg("KEYBOARD FLYING" if kb_mode else "JOYSTICK")
                    elif k == pg.K_SPACE:
                        kb_armed = not kb_armed
                    elif k == pg.K_F12 and logger:
                        pixels = ctx.screen.read(viewport=(0, 0, rend.w, rend.h), components=3, alignment=1)
                        pg.image.save(pg.image.frombytes(pixels, (rend.w, rend.h), "RGB", True),
                                      str(logger.dir / f"shot_{frame:06d}.png"))
            if not running:
                break
            now = time.perf_counter()
            # ModernGL can cache the default framebuffer's creation size during
            # fullscreen transitions. SDL's window size is authoritative.
            drawable_size = tuple(pg.display.get_window_size())
            if rend.resize(drawable_size):
                W, H = drawable_size
                last_osd = 0.0
                surface_size = tuple(pg.display.get_surface().get_size())
                print(f"[fpv] display resize: window={drawable_size} "
                      f"surface={surface_size} moderngl={tuple(ctx.screen.size)}")
            frame_dt = now - last
            last = now
            # -------- input (sampled right before physics)
            arm_hint = "PRESS SPACE"
            if track_menu_open:
                st.roll = st.pitch = st.yaw = st.throttle = 0.0
                want_arm = False
            elif a.script:
                want_arm = scripted_takeoff(now - t_start, st)
            elif kb_mode:
                kb.read(pg, st, frame_dt)
                if joy.connected:
                    joy_st = joy.read(inputs.Sticks())
                    st.raw = joy_st.raw
                want_arm = kb_armed
            else:
                joy.read(st)
                sw = joy.switch("arm", st)
                want_arm = kb_armed if sw is None else sw
                arm_hint = "PRESS SPACE" if sw is None else "USE ARM SWITCH"
                pp = joy.switch_pos("preset", st)
                if pp is not None and pp != last_preset_pos:
                    sim.set_preset(sim.preset_cycle[min(pp, len(sim.preset_cycle) - 1)])
                    last_preset_pos = pp
                rs = joy.switch("reset", st)
                if rs and not last_reset_sw:
                    sim.reset("switch")
                last_reset_sw = bool(rs)
            sim.set_armed(bool(want_arm), st.throttle, arm_hint=arm_hint)
            # -------- physics: fixed 1 kHz steps covering the elapsed wall time
            if track_menu_open:
                accum = 0.0
            else:
                accum += frame_dt
                if accum > 0.05:
                    lag_events += 1
                    if logger:
                        logger.event("lag", t=sim.t, dropped_s=accum - 0.05)
                    accum = 0.05
                n = int(accum / dt)
                accum -= n * dt
                if n:
                    sim.advance(n, st)
            # -------- render
            mode = cam_modes[cam_idx]
            hfov = render.spec_hfov_deg(cfg["camera"]) if fov_mode == "spec" else float(cfg["camera"]["hfov_wide_deg"])
            eid, target = sim.next_target()
            if now - last_osd >= osd_period:
                last_osd = now
                s = sim.s
                lt = sim.laps
                cur = f"{sim.t - lt.lap_start:6.2f}" if lt.started else "  --  "
                last_lap = f"{lt.laps[-1]['time_s']:.2f}{'' if lt.laps[-1]['valid'] else '*'}" if lt.laps else "--"
                best = lt.best_lap()
                dist = math.dist(s[0:3], target)
                mr = sim.profile.max_rates_dps()
                center = None
                if sim.crashed_at is not None:
                    center = "CRASH"
                rend.update_osd({
                    "tl": [f"LAP  {cur}", f"LAST {last_lap}", f"BEST {best['time_s']:.2f}" if best else "BEST --",
                           f"LAPS {len(lt.valid_laps())}/{len(lt.laps)}"],
                    "tr": [f"{fps:4.0f} FPS", f"RATES {sim.preset} {mr[0]:.0f}/{mr[1]:.0f}/{mr[2]:.0f}",
                           f"TILT {sim.tilt_deg:.0f}  FOV {hfov:.0f} {fov_mode}", f"CAM {mode}"],
                    "bl": ["ARMED" if sim.armed else "DISARMED", f"THR {st.throttle * 100:3.0f}%",
                           f"{np.linalg.norm(s[3:6]):4.1f} m/s  ALT {s[2]:4.1f}"]
                          + (["SCRIPT"] if a.script else (["KEYBOARD"] if kb_mode else [])),
                    "top": f"NEXT {eid if eid != sim.env.element_ids[0] else 'START LINE'}  {dist:4.1f} m",
                    "center": center, "msgs": sim.active_messages(), "thr": st.throttle,
                    "help_columns": HELP_COLUMNS if show_help else None,
                    "track_menu": ({"items": track_presets, "selected": track_selection,
                                    "current": current_track}
                                   if track_menu_open else None)})
            cam_p, cam_q, vfov = rend.render(ctx.screen, sim.s, mode, sim.tilt_deg, hfov, target, osd=True)
            if logger:
                s = sim.s
                logger.frame([sim.t, frame, cam_idx, *cam_p, *cam_q, *s[0:3], *s[6:10],
                              sim.tilt_deg, hfov, vfov, time.time()])
            pg.display.flip()
            frame += 1
            fps = 0.95 * fps + 0.05 / max(frame_dt, 1e-6) if frame > 1 else 0.0
            if a.screenshot_every and logger and now - last_shot >= a.screenshot_every:
                last_shot = now
                pixels = ctx.screen.read(viewport=(0, 0, rend.w, rend.h), components=3, alignment=1)
                img = pg.image.frombytes(pixels, (rend.w, rend.h), "RGB", True)
                pg.image.save(img, str(logger.dir / f"shot_{frame:06d}_{mode}.png"))
                cam_idx = (cam_idx + 1) % len(cam_modes) if a.script else cam_idx
            if a.duration and now - t_start >= a.duration:
                running = False
    finally:
        wall = time.perf_counter() - t_start
        sim.close()
        pg.quit()
        print(f"[fpv] {frame} frames in {wall:.1f}s = {frame / max(wall, 1e-9):.0f} fps; "
              f"sim time {sim.t:.2f}s ({sim.t / max(wall, 1e-9):.3f}x real time); lag events {lag_events}")
        print(f"[fpv] laps {len(sim.laps.laps)} crashes {sim.n_crashes} resets {sim.n_resets}")
        if logger:
            print(f"[fpv] log: {logger.dir}")
    if next_track is not None:
        new_args = argv_for_track(sys.argv[1:], next_track)
        print(f"[fpv] switching track to {next_track}")
        command = [sys.executable, "-m", "fpvsim", *new_args]
        if os.name == "nt":
            subprocess.Popen(command, cwd=HERE)
            return
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
