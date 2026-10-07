"""Betaflight stick -> rate curves, deadband and throttle curve.

Ported from Betaflight 4.x `src/main/fc/rc.c`:
  applyBetaflightRates(), applyActualRates(), processRcCommand() deadband,
  and the throttle lookup table in `generateThrottleCurve()`.
Stick input here is normalised to [-1, 1] (throttle [0, 1]).
"""
from __future__ import annotations

import math

RC_RATE_INCREMENTAL = 14.54   # BF: extra slope for rc_rate above 2.0
RATE_LIMIT_DPS = 1998.0       # BF rate_limit default


def power3(x):
    return x * x * x


def power5(x):
    return x * x * x * x * x


def apply_deadband(stick: float, deadband: float) -> float:
    """BF deadband: rcCommand in [-500,500] minus deadband, divided by (500-deadband)."""
    cmd = max(-500.0, min(500.0, stick * 500.0))
    if abs(cmd) <= deadband:
        return 0.0
    cmd -= math.copysign(deadband, cmd)
    return cmd / (500.0 - deadband)


def betaflight_rate(x: float, rc_rate: float, super_rate: float, expo: float) -> float:
    """BF 'Betaflight' rates. rc_rate/super_rate/expo as fractions (BF value / 100)."""
    ax = abs(x)
    if expo:
        x = x * power3(ax) * expo + x * (1.0 - expo)
    rc = rc_rate
    if rc > 2.0:
        rc += RC_RATE_INCREMENTAL * (rc - 2.0)
    angle_rate = 200.0 * rc * x
    if super_rate:
        sf = 1.0 / min(max(1.0 - ax * super_rate, 0.01), 1.0)
        angle_rate *= sf
    return angle_rate


def actual_rate(x: float, center_dps: float, max_dps: float, expo: float) -> float:
    """BF 'Actual' rates (center sensitivity, max rate, expo)."""
    ax = abs(x)
    expof = ax * (power5(x) * expo + x * (1.0 - expo))
    stick_movement = max(0.0, max_dps - center_dps)
    return x * center_dps + stick_movement * expof


def axis_rate_dps(x: float, axis_cfg: dict, rtype: str) -> float:
    if rtype == "betaflight":
        r = betaflight_rate(x, float(axis_cfg["rc_rate"]), float(axis_cfg["super_rate"]),
                            float(axis_cfg.get("expo", 0.0)))
    elif rtype == "actual":
        r = actual_rate(x, float(axis_cfg["center_dps"]), float(axis_cfg["max_dps"]),
                        float(axis_cfg.get("expo", 0.0)))
    else:
        raise ValueError(f"unknown rates type {rtype!r}")
    return max(-RATE_LIMIT_DPS, min(RATE_LIMIT_DPS, r))


def max_rate_dps(axis_cfg: dict, rtype: str) -> float:
    return axis_rate_dps(1.0, axis_cfg, rtype)


def throttle_curve(x: float, mid: float, expo: float) -> float:
    """BF thr_mid / thr_expo, continuous form of the lookup table. x, out in [0,1]."""
    x = max(0.0, min(1.0, x))
    tmp = x - mid
    if tmp > 0:
        y = 1.0 - mid
    elif tmp < 0:
        y = mid
    else:
        return mid
    return mid + tmp * ((1.0 - expo) + expo * (tmp * tmp) / (y * y))


class RateProfile:
    """Maps normalised sticks to body-rate setpoints for one preset."""

    def __init__(self, name: str, preset: dict, rc_deadband: float, yaw_deadband: float,
                 thr_mid: float, thr_expo: float):
        self.name = name
        self.type = preset["type"]
        self.axes = [preset["roll"], preset["pitch"], preset["yaw"]]
        self.deadbands = [rc_deadband, rc_deadband, yaw_deadband]
        self.thr_mid, self.thr_expo = thr_mid, thr_expo

    def setpoints_dps(self, roll: float, pitch: float, yaw: float):
        """Stick convention: roll + right, pitch + forward, yaw + right. Returns dps."""
        out = []
        for x, cfg, db in zip((roll, pitch, yaw), self.axes, self.deadbands):
            out.append(axis_rate_dps(apply_deadband(x, db), cfg, self.type))
        return out

    def throttle(self, t: float) -> float:
        return throttle_curve(t, self.thr_mid, self.thr_expo)

    def max_rates_dps(self):
        return [max_rate_dps(c, self.type) for c in self.axes]


def stick_to_body_rates(sp_dps):
    """Stick-frame rates (roll right, pitch forward=nose down, yaw right) -> body FLU rad/s.

    Body frame is FLU (x forward, y left, z up). Right-hand rule:
      roll right  = +wx,   nose down = +wy,   nose right = -wz.
    """
    k = math.pi / 180.0
    return (sp_dps[0] * k, sp_dps[1] * k, -sp_dps[2] * k)
