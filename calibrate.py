#!/usr/bin/env python3
"""Interactive transmitter calibration -> config/calibration.yaml

    .venv/bin/python calibrate.py
    ... calibrate.py --show        # live raw axis/button values only

Detects, by asking you to move one control at a time:
  stick axis -> channel mapping, inversion, min / centre / max  (Mode 2 prompts)
  ARM switch, RATE PRESET switch (2/3 position), optional RESET switch.
"""
from __future__ import annotations

import argparse
import os
import select
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
import pygame as pg  # noqa: E402


def open_js(name_hint):
    pg.init()
    pg.joystick.init()
    n = pg.joystick.get_count()
    if n == 0:
        sys.exit("No joystick found. Plug the transmitter in USB joystick (HID) mode.")
    for i in range(n):
        j = pg.joystick.Joystick(i)
        if name_hint.lower() in j.get_name().lower():
            return j
    return pg.joystick.Joystick(0)


def read(js):
    pg.event.pump()
    return ([js.get_axis(i) for i in range(js.get_numaxes())],
            [js.get_button(i) for i in range(js.get_numbuttons())])


def sample_until_enter(js, prompt, live=True):
    """Sample continuously until Enter. Returns (typed, samples[list of (axes, buttons)])."""
    print("\n" + prompt)
    print("  ...press Enter when done (type s + Enter to skip)", flush=True)
    samples = []
    while True:
        ax, bt = read(js)
        samples.append((ax, bt))
        if live and len(samples) % 10 == 0:
            s = " ".join(f"{v:+.2f}" for v in ax)
            b = "".join(str(x) for x in bt[:16])
            print(f"\r  axes [{s}] btn {b}   ", end="", flush=True)
        r, _, _ = select.select([sys.stdin], [], [], 0.005)
        if r:
            typed = sys.stdin.readline().strip().lower()
            print()
            return typed, samples


def ranges(samples, na):
    lo = [min(s[0][i] for s in samples) for i in range(na)]
    hi = [max(s[0][i] for s in samples) for i in range(na)]
    return lo, hi


def show(js):
    print(f"{js.get_name()}: {js.get_numaxes()} axes, {js.get_numbuttons()} buttons. Ctrl-C to stop.")
    try:
        while True:
            ax, bt = read(js)
            print("\r" + " ".join(f"a{i}:{v:+.3f}" for i, v in enumerate(ax))
                  + "  b:" + "".join(str(x) for x in bt[:16]), end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "config" / "calibration.yaml"))
    ap.add_argument("--device", default="Zorro")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    js = open_js(a.device)
    if a.show:
        return show(js)
    na, nb = js.get_numaxes(), js.get_numbuttons()
    print(f"Found: {js.get_name()}  ({na} axes, {nb} buttons)")
    print("Mode 2 assumed: LEFT stick = throttle (up/down) + yaw (left/right),"
          " RIGHT stick = pitch (up/down) + roll (left/right).")

    _, rest_s = sample_until_enter(js, "STEP 1: centre roll/pitch/yaw, throttle fully DOWN, "
                                       "all switches in their normal (DISARMED) position.")
    rest = rest_s[-1][0]
    rest_btn = rest_s[-1][1]

    axes_cfg, used = {}, set()
    plan = [("throttle", "the THROTTLE (left stick up/down)", "fully UP"),
            ("yaw", "YAW (left stick left/right)", "fully RIGHT"),
            ("pitch", "PITCH (right stick up/down)", "fully FORWARD/UP"),
            ("roll", "ROLL (right stick left/right)", "fully RIGHT")]
    for name, desc, hold in plan:
        _, smp = sample_until_enter(js, f"Move ONLY {desc} through its full range a few times.")
        lo, hi = ranges(smp, na)
        cand = sorted((hi[i] - lo[i], i) for i in range(na) if i not in used)
        rng_, ax = cand[-1]
        if rng_ < 0.5:
            print(f"  WARNING: largest movement only {rng_:.2f} on axis {ax}; mapping may be wrong")
        used.add(ax)
        _, hold_s = sample_until_enter(js, f"Hold {desc} {hold}.", live=False)
        v = hold_s[-1][0][ax]
        if name == "throttle":
            invert = v < (lo[ax] + hi[ax]) / 2
            axes_cfg[name] = {"axis": ax, "min": round(lo[ax], 4), "max": round(hi[ax], 4), "invert": bool(invert)}
        else:
            invert = v < rest[ax]
            axes_cfg[name] = {"axis": ax, "min": round(lo[ax], 4), "center": round(rest[ax], 4),
                              "max": round(hi[ax], 4), "invert": bool(invert)}
        print(f"  {name}: axis {ax}  range [{lo[ax]:+.3f}, {hi[ax]:+.3f}]  invert={invert}")

    switches = {}

    def detect_switch(prompt):
        typed, smp = sample_until_enter(js, prompt, live=False)
        if typed == "s":
            return None, smp
        ax, bt = smp[-1]
        best = (0.0, None)
        for i in range(na):
            if i in used:
                continue
            d = abs(ax[i] - rest[i])
            if d > best[0]:
                best = (d, ("axis", i))
        for i in range(nb):
            if bt[i] != rest_btn[i] and 1.0 > best[0] * 0.5:
                best = max(best, (1.0, ("button", i)))
        if best[1] is None or best[0] < 0.3:
            print("  no switch change detected; skipped")
            return None, smp
        return best[1], smp

    found, smp = detect_switch("Flip the switch you want for ARM into the ARMED position (leave it there).")
    if found:
        kind, idx = found
        if kind == "axis":
            on = smp[-1][0][idx]
            invert = on < rest[idx]
            r_t, on_t = (-rest[idx], -on) if invert else (rest[idx], on)
            switches["arm"] = {"kind": "axis", "index": idx, "invert": bool(invert),
                               "threshold": round((r_t + on_t) / 2, 3)}
        else:
            switches["arm"] = {"kind": "button", "index": idx, "invert": bool(rest_btn[idx]), "threshold": 0.0}
        print(f"  arm: {switches['arm']}")
        if kind == "axis":
            used.add(idx)
        sample_until_enter(js, "Now put the ARM switch back to DISARMED.", live=False)

    typed, smp = sample_until_enter(js, "RATE PRESET switch: move it through ALL its positions, "
                                        "ending in the first (low) position.")
    if typed != "s":
        lo, hi = ranges(smp, na)
        cand = sorted((hi[i] - lo[i], i) for i in range(na) if i not in used)
        if cand and cand[-1][0] > 0.5:
            idx = cand[-1][1]
            vals = sorted({round(s[0][idx] * 2) / 2 for s in smp})
            clusters = [v for v in vals if any(abs(s[0][idx] - v) < 0.15 for s in smp)]
            positions = max(2, min(3, len(clusters)))
            invert = smp[-1][0][idx] > (lo[idx] + hi[idx]) / 2
            switches["preset"] = {"kind": "axis", "index": idx, "positions": positions, "invert": bool(invert)}
            used.add(idx)
            print(f"  preset: {switches['preset']}  (positions map to rates.preset_cycle)")
        else:
            print("  no axis moved; skipped")

    found, smp = detect_switch("OPTIONAL reset: hold the switch/button you want for RESET (or s to skip).")
    if found:
        kind, idx = found
        if kind == "axis":
            on = smp[-1][0][idx]
            invert = on < rest[idx]
            r_t, on_t = (-rest[idx], -on) if invert else (rest[idx], on)
            switches["reset"] = {"kind": "axis", "index": idx, "invert": bool(invert),
                                 "threshold": round((r_t + on_t) / 2, 3)}
        else:
            switches["reset"] = {"kind": "button", "index": idx, "invert": bool(rest_btn[idx]), "threshold": 0.0}
        print(f"  reset: {switches['reset']}")

    out = {"device_name": js.get_name(), "verified": True, "mode": 2,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "axes": axes_cfg, "switches": switches}
    Path(a.out).write_text("# written by calibrate.py\n" + yaml.safe_dump(out, sort_keys=False))
    print(f"\nWrote {a.out}")

    sys.path.insert(0, str(HERE))
    from fpvsim import inputs
    joy = inputs.JoystickInput(out, pg)
    st = inputs.Sticks()
    print("Live check for 8 s -- roll right, pitch forward and yaw right should read POSITIVE:")
    t0 = time.time()
    while time.time() - t0 < 8:
        pg.event.pump()
        joy.read(st)
        arm = joy.switch("arm", st)
        pre = joy.switch_pos("preset", st)
        print(f"\r  roll {st.roll:+.2f} pitch {st.pitch:+.2f} yaw {st.yaw:+.2f} thr {st.throttle:.2f}"
              f"  arm {arm} preset {pre}   ", end="", flush=True)
        time.sleep(0.03)
    print()


if __name__ == "__main__":
    main()
