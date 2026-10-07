"""RC transmitter (USB joystick) input via pygame/SDL, driven by a calibration YAML.

Normalised stick convention used everywhere in the sim:
  roll  +1 = stick right      pitch +1 = stick forward (nose down)
  yaw   +1 = stick right      throttle 0 = low, 1 = high
"""
from __future__ import annotations

NRAW = 8


def norm_bipolar(raw, a):
    c, lo, hi = float(a.get("center", 0.0)), float(a.get("min", -1.0)), float(a.get("max", 1.0))
    if raw >= c:
        v = (raw - c) / max(hi - c, 1e-6)
    else:
        v = (raw - c) / max(c - lo, 1e-6)
    v = max(-1.0, min(1.0, v))
    return -v if a.get("invert", False) else v


def norm_unipolar(raw, a):
    lo, hi = float(a.get("min", -1.0)), float(a.get("max", 1.0))
    v = (raw - lo) / max(hi - lo, 1e-6)
    v = max(0.0, min(1.0, v))
    return 1.0 - v if a.get("invert", False) else v


def switch_value(raw_axes, buttons, sw):
    """Return a value in [-1, 1] for an axis switch or {-1, 1} for a button."""
    if not sw:
        return None
    idx = int(sw["index"])
    if sw.get("kind", "axis") == "button":
        v = 1.0 if (idx < len(buttons) and buttons[idx]) else -1.0
    else:
        v = raw_axes[idx] if idx < len(raw_axes) else -1.0
    return -v if sw.get("invert", False) else v


def switch_on(raw_axes, buttons, sw):
    v = switch_value(raw_axes, buttons, sw)
    return v is not None and v > float(sw.get("threshold", 0.0))


def switch_position(raw_axes, buttons, sw):
    """Discrete position 0..positions-1 (e.g. 3-position switch -> 0,1,2)."""
    v = switch_value(raw_axes, buttons, sw)
    if v is None:
        return None
    n = int(sw.get("positions", 2))
    k = int((v + 1.0) / 2.0 * n)
    return max(0, min(n - 1, k))


class Sticks:
    __slots__ = ("roll", "pitch", "yaw", "throttle", "raw", "buttons")

    def __init__(self):
        self.roll = self.pitch = self.yaw = self.throttle = 0.0
        self.raw = [0.0] * NRAW
        self.buttons = []


class JoystickInput:
    def __init__(self, calib: dict, pygame):
        self.pg = pygame
        self.calib = calib
        self.js = None
        pygame.joystick.init()
        want = str(calib.get("device_name", "")).lower()
        for i in range(pygame.joystick.get_count()):
            j = pygame.joystick.Joystick(i)
            if not want or want in j.get_name().lower():
                self.js = j
                break
        if self.js is None and pygame.joystick.get_count() > 0:
            self.js = pygame.joystick.Joystick(0)
        self.name = self.js.get_name() if self.js else None

    @property
    def connected(self):
        return self.js is not None

    def read(self, out: Sticks) -> Sticks:
        if self.js is None:
            return out
        na = self.js.get_numaxes()
        raw = [self.js.get_axis(i) for i in range(na)]
        out.raw = (raw + [0.0] * NRAW)[:NRAW]
        out.buttons = [self.js.get_button(i) for i in range(self.js.get_numbuttons())]
        ax = self.calib["axes"]
        out.roll = norm_bipolar(raw[ax["roll"]["axis"]], ax["roll"])
        out.pitch = norm_bipolar(raw[ax["pitch"]["axis"]], ax["pitch"])
        out.yaw = norm_bipolar(raw[ax["yaw"]["axis"]], ax["yaw"])
        out.throttle = norm_unipolar(raw[ax["throttle"]["axis"]], ax["throttle"])
        return out

    def switch(self, name, sticks: Sticks):
        sw = (self.calib.get("switches") or {}).get(name)
        return switch_on(sticks.raw, sticks.buttons, sw) if sw else None

    def switch_pos(self, name, sticks: Sticks):
        sw = (self.calib.get("switches") or {}).get(name)
        return switch_position(sticks.raw, sticks.buttons, sw) if sw else None
