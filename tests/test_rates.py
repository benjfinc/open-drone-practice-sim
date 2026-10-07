"""Rate curves against Betaflight's rc.c formulas (independent re-implementation in BF units)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpvsim import config as C  # noqa: E402
from fpvsim.rates import (RateProfile, actual_rate, apply_deadband, betaflight_rate,  # noqa: E402
                          stick_to_body_rates, throttle_curve)


# ---- literal transcription of Betaflight 4.x rc.c, integer config units ----
def bf_c_betaflight(rcCommandf, rcRates, rates, rcExpo):
    rcCommandfAbs = abs(rcCommandf)
    if rcExpo:
        expof = rcExpo / 100.0
        rcCommandf = rcCommandf * rcCommandfAbs ** 3 * expof + rcCommandf * (1 - expof)
    rcRate = rcRates / 100.0
    if rcRate > 2.0:
        rcRate += 14.54 * (rcRate - 2.0)
    angleRate = 200.0 * rcRate * rcCommandf
    if rates:
        rcSuperfactor = 1.0 / min(max(1.0 - (rcCommandfAbs * (rates / 100.0)), 0.01), 1.00)
        angleRate *= rcSuperfactor
    return angleRate


def bf_c_actual(rcCommandf, rcRates, rates, rcExpo):
    rcCommandfAbs = abs(rcCommandf)
    expof = rcExpo / 100.0
    expof = rcCommandfAbs * (rcCommandf ** 5 * expof + rcCommandf * (1 - expof))
    centerSensitivity = rcRates * 10.0
    stickMovement = max(0, rates * 10.0 - centerSensitivity)
    return rcCommandf * centerSensitivity + stickMovement * expof


XS = [-1.0, -0.73, -0.5, -0.2, -0.01, 0.0, 0.05, 0.33, 0.5, 0.8, 0.99, 1.0]


@pytest.mark.parametrize("rc,sr,ex", [(100, 70, 0), (120, 75, 20), (250, 0, 0), (180, 60, 50), (50, 90, 0)])
def test_betaflight_rates_match_rc_c(rc, sr, ex):
    for x in XS:
        assert betaflight_rate(x, rc / 100, sr / 100, ex / 100) == pytest.approx(bf_c_betaflight(x, rc, sr, ex), abs=1e-9)


@pytest.mark.parametrize("c,m,e", [(7, 67, 0), (20, 67, 54), (25, 100, 50), (15, 50, 30)])
def test_actual_rates_match_rc_c(c, m, e):
    # BF stores Actual centre/max in units of 10 deg/s
    for x in XS:
        assert actual_rate(x, c * 10, m * 10, e / 100) == pytest.approx(bf_c_actual(x, c, m, e), abs=1e-9)


def test_published_numbers():
    # rc_rate 1.00 / super 0.70 -> 666.7 deg/s max (BF Configurator)
    assert betaflight_rate(1.0, 1.0, 0.70, 0.0) == pytest.approx(666.67, abs=0.01)
    assert betaflight_rate(0.5, 1.0, 0.70, 0.0) == pytest.approx(100 / 0.65, abs=1e-9)
    # rc_rate above 2.0 uses the 14.54 incremental slope
    assert betaflight_rate(1.0, 2.5, 0.0, 0.0) == pytest.approx(200 * (2.5 + 14.54 * 0.5))
    # Actual 70 / 670 / 0 (BF 4.3 defaults): max 670, half stick 35 + 600 * 0.25
    assert actual_rate(1.0, 70, 670, 0.0) == pytest.approx(670)
    assert actual_rate(0.5, 70, 670, 0.0) == pytest.approx(185.0)
    assert actual_rate(0.5, 70, 670, 0.54) == pytest.approx(35 + 600 * 0.5 * (0.5 ** 5 * 0.54 + 0.5 * 0.46))
    # centre sensitivity is the slope at zero
    assert actual_rate(1e-4, 200, 670, 0.54) / 1e-4 == pytest.approx(200, rel=1e-3)


def test_odd_symmetry():
    for x in XS:
        assert actual_rate(-x, 200, 670, 0.5) == pytest.approx(-actual_rate(x, 200, 670, 0.5))
        assert betaflight_rate(-x, 1.2, 0.7, 0.3) == pytest.approx(-betaflight_rate(x, 1.2, 0.7, 0.3))


def test_deadband():
    assert apply_deadband(4.0 / 500, 5) == 0.0
    assert apply_deadband(-5.0 / 500, 5) == 0.0
    assert apply_deadband(1.0, 5) == pytest.approx(1.0)
    assert apply_deadband(-1.0, 5) == pytest.approx(-1.0)
    assert apply_deadband(255 / 500, 5) == pytest.approx(250 / 495)


def test_throttle_curve():
    for x in (0, 0.1, 0.37, 0.5, 0.9, 1.0):
        assert throttle_curve(x, 0.5, 0.0) == pytest.approx(x)
    for mid, ex in ((0.5, 1.0), (0.3, 0.6), (0.7, 0.4)):
        assert throttle_curve(0.0, mid, ex) == pytest.approx(0.0)
        assert throttle_curve(1.0, mid, ex) == pytest.approx(1.0)
        assert throttle_curve(mid, mid, ex) == pytest.approx(mid)
        ys = [throttle_curve(i / 100, mid, ex) for i in range(101)]
        assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))
    # expo flattens around mid
    assert abs(throttle_curve(0.55, 0.5, 1.0) - 0.5) < abs(0.55 - 0.5)


def test_presets_in_config():
    cfg = C.load_config()
    r = cfg["rates"]
    prof = {n: RateProfile(n, p, r["rc_deadband"], r["yaw_deadband"], 0.5, 0.0) for n, p in r["presets"].items()}
    assert prof["default"].max_rates_dps()[0] == pytest.approx(670)
    assert prof["pro"].max_rates_dps()[0] == pytest.approx(670)
    assert prof["max"].max_rates_dps()[0] >= 1000
    assert prof["bf_classic"].max_rates_dps()[0] == pytest.approx(666.67, abs=0.01)


def test_stick_to_body_signs():
    wx, wy, wz = stick_to_body_rates((100, 100, 100))
    assert wx > 0 and wy > 0 and wz < 0
    assert wx == pytest.approx(math.radians(100))
