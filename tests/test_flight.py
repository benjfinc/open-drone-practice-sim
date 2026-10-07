"""Scripted-stick flight tests on the full physics + rate loop (no human, no window)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpvsim import config as C  # noqa: E402
from fpvsim import dynamics as D  # noqa: E402
from fpvsim.rates import RateProfile, stick_to_body_rates  # noqa: E402

FAR = np.zeros((0, 5))
NOE = np.zeros((0, 8))
REPORT = {}


@pytest.fixture(scope="module")
def cfg():
    return C.load_config()


@pytest.fixture(scope="module")
def P(cfg):
    # free air: these tests fly at 50 m, above the track ceiling
    return D.build_params(C.deep_merge(cfg, {"track": {"ceiling_height_m": None}}))


class Q:
    def __init__(self, P, pos=(0, 0, 50.0), yaw=0.0, hover=True):
        self.P = P
        self.s = D.new_state(pos, yaw)
        self.c = D.new_ctrl()
        self.hov = D.hover_throttle_stick(P)
        if hover:                                         # steady hover: motors already spun up
            self.s[D.S_M:D.S_M + 4] = P[D.P_IDLE] + (1 - P[D.P_IDLE]) * self.hov
        self.L, self.T, self.xb = np.zeros((1, D.NLOG), np.float32), np.zeros(1), np.zeros((4, 5))
        self.dp = np.zeros(0)

    def run(self, sp, thr, n):
        return D.step_n(self.s, self.c, self.P, FAR, NOE, self.dp, np.asarray(sp, float), thr, True,
                        n, 0.0, self.L, self.T, -1, self.xb)


def test_hover_throttle_holds_altitude(P):
    q = Q(P)
    q.run((0, 0, 0), q.hov, 5000)
    dz = q.s[2] - 50.0
    REPORT["hover_dz_5s_m"] = dz
    assert abs(dz) < 0.02 and abs(q.s[5]) < 0.01
    assert np.abs(q.s[D.S_W:D.S_W + 3]).max() < 1e-6
    assert 0.2 < q.hov < 0.6                                  # hover stick in a plausible band


def step_response(P, axis, rate_dps, ms=400):
    q = Q(P)
    sp = [0.0, 0.0, 0.0]
    sp[axis] = math.radians(rate_dps)
    w = []
    for _ in range(ms):
        q.run(sp, q.hov, 1)
        w.append(q.s[D.S_W + axis])
    w = np.array(w) / sp[axis]
    t90 = int(np.argmax(w >= 0.9)) + 1 if (w >= 0.9).any() else None
    return t90, w.max(), w[-1]


@pytest.mark.parametrize("axis,limit_ms", [(0, 120), (1, 120), (2, 250)])
def test_full_stick_reaches_max_rate(cfg, P, axis, limit_ms):
    r = cfg["rates"]
    prof = RateProfile("default", r["presets"]["default"], r["rc_deadband"], r["yaw_deadband"], 0.5, 0.0)
    sticks = [0.0, 0.0, 0.0]
    sticks[axis] = 1.0
    sp_body = stick_to_body_rates(prof.setpoints_dps(*sticks))
    target = abs(math.degrees(sp_body[axis]))
    assert target == pytest.approx(670)
    t90, peak, final = step_response(P, axis, math.degrees(sp_body[axis]))
    tau_ms = cfg["physics"]["motor_tau_up_s"] * 1000
    REPORT[f"axis{axis}_t90_ms"] = t90
    REPORT[f"axis{axis}_overshoot"] = peak - 1
    print(f"\naxis {axis}: t90 {t90} ms (motor tau {tau_ms:.0f} ms), overshoot {100 * (peak - 1):.1f}%, "
          f"final {final:.3f}")
    assert t90 is not None and t90 <= limit_ms
    assert abs(final - 1) < 0.06
    assert peak < 1.25


def test_centered_sticks_are_stable(P):
    rng = np.random.default_rng(1)
    q = Q(P, pos=(0, 0, 500.0))
    for _ in range(150):
        q.run(rng.uniform(-1, 1, 3) * math.radians(900), rng.uniform(0.2, 0.8), 20)
    # After aggressive saturating inputs the I-term (Ki/Kp ~ 2/s) leaves a slow
    # tail. Require it to DECAY (no divergence, no limit cycle) and to be small
    # by 3 s. Numbers are reported, not tuned to pass.
    ws, drifts = [], []
    for sec in range(3):
        q0 = q.s[D.S_Q:D.S_Q + 4].copy()
        q.run((0, 0, 0), 0.5, 1000)
        ws.append(float(np.abs(q.s[D.S_W:D.S_W + 3]).max()))
        drifts.append(2 * math.acos(min(1.0, abs(float(q0 @ q.s[D.S_Q:D.S_Q + 4])))))
    REPORT["centered_wmax_at_1_2_3s"] = ws
    REPORT["attitude_drift_per_second_rad"] = drifts
    print(f"\ncentered: max|w| at 1/2/3 s {ws}; attitude change per second {drifts}")
    assert not np.isnan(q.s).any()
    assert ws[0] < 0.1 and ws[1] < ws[0] and ws[2] < ws[1] and ws[2] < 0.01
    assert drifts[2] < drifts[0] and drifts[2] < 0.005


def test_stick_directions_move_the_drone_correctly(P):
    # pitch forward -> nose down -> accelerates FORWARD along heading (yaw 0.7 rad)
    yaw = 0.7
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0])
    for axis, stick_dir, expect in [(1, fwd, 1), (0, left, -1)]:
        q = Q(P, yaw=yaw)
        sp = [0, 0, 0]
        sp[axis] = math.radians(200)
        q.run(sp, q.hov, 100)                       # tilt ~20 deg
        q.run((0, 0, 0), q.hov + 0.05, 800)
        v = q.s[D.S_V:D.S_V + 3]
        assert expect * (v @ stick_dir) > 0.5, (axis, v)
    q = Q(P, yaw=yaw)
    sp = stick_to_body_rates((0, 0, 200))           # yaw stick right
    q.run(sp, q.hov, 300)
    R = D.quat_to_R_np(q.s[D.S_Q:D.S_Q + 4])
    heading = math.atan2(R[1, 0], R[0, 0])
    assert heading < yaw - 0.3                      # nose turned right (clockwise from above)


def test_physics_rate_budget(P):
    import time
    q = Q(P, pos=(0, 0, 1e6))
    q.run((1, 0.5, 0.2), 0.6, 1000)
    n, t0 = 200000, time.perf_counter()
    q.run((1, 0.5, 0.2), 0.6, n)
    hz = n / (time.perf_counter() - t0)
    REPORT["physics_ticks_per_s"] = hz
    print(f"\nphysics ticks/s: {hz:.0f}")
    assert hz > 20000                              # >=20x the 1 kHz requirement


def teardown_module(module):
    print("\nREPORT", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in REPORT.items()})


def test_ceiling_stops_the_climb(cfg, P):
    """Full throttle from below must stop at ceiling_height_m, and not crash by default."""
    ch = cfg["track"]["ceiling_height_m"]
    assert ch is not None and not cfg["track"]["ceiling_crash"]
    Pc = D.build_params(cfg)
    D.configure_venue_bounds(Pc, (-10.0, 10.0, -10.0, 10.0), cfg)
    q = Q(Pc, pos=(0, 0, ch - 3.0))
    ev = q.run((0, 0, 0), 1.0, 4000)[1]
    REPORT["ceiling_z_after_full_throttle"] = float(q.s[2])
    assert ev == 0                                        # sliding along it is not a crash
    assert ch - q.P[D.P_RAD] - 1e-6 <= q.s[2] <= ch         # parked just under it
    assert q.s[5] <= 0.01                                 # no upward velocity left


def test_ceiling_can_be_disabled_and_can_crash(cfg):
    off = C.deep_merge(cfg, {"track": {"ceiling_height_m": None}})
    Po = D.build_params(off)
    D.configure_venue_bounds(Po, (-10.0, 10.0, -10.0, 10.0), off)
    q = Q(Po, pos=(0, 0, cfg["track"]["ceiling_height_m"] - 3.0))
    q.run((0, 0, 0), 1.0, 4000)
    assert q.s[2] > cfg["track"]["ceiling_height_m"]      # climbs straight through

    hard = C.deep_merge(cfg, {"track": {"ceiling_crash": True}})
    Ph = D.build_params(hard)
    D.configure_venue_bounds(Ph, (-10.0, 10.0, -10.0, 10.0), hard)
    q = Q(Ph, pos=(0, 0, cfg["track"]["ceiling_height_m"] - 12.0))
    assert q.run((0, 0, 0), 1.0, 6000)[1] == 3            # fast hit = ceiling crash
