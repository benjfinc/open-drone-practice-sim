import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpvsim import config as C  # noqa: E402
from fpvsim import dynamics as D  # noqa: E402


@pytest.fixture(scope="module")
def P():
    return D.build_params(C.load_config())


def torques(P, f):
    T = P[D.P_TMAX] * np.asarray(f)
    mx, my, ms = P[D.P_MX0:D.P_MX0 + 4], P[D.P_MY0:D.P_MY0 + 4], P[D.P_MS0:D.P_MS0 + 4]
    return np.array([my @ T, -(mx @ T), P[D.P_YAWK] * (ms @ T)])


def run_mix(P, thr, t, airmode=True):
    out = np.zeros(4)
    D.mix(P, thr, t[0], t[1], t[2], airmode, out)
    return out


def test_zero_demand_equal_motors(P):
    f = run_mix(P, 0.4, (0, 0, 0))
    c = P[D.P_IDLE] + (1 - P[D.P_IDLE]) * 0.4
    assert np.allclose(f, c * c)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_allocation_is_exact_when_unsaturated(P, axis):
    t = np.zeros(3)
    t[axis] = 0.05
    f = run_mix(P, 0.5, t)
    got = torques(P, f)
    assert got == pytest.approx(t, abs=1e-9)
    assert (f.sum() - 4 * run_mix(P, 0.5, (0, 0, 0))[0]) == pytest.approx(0, abs=1e-12)


def test_motor_layout_signs(P):
    # order: 1 RR, 2 FR, 3 RL, 4 FL
    f = run_mix(P, 0.5, (0.05, 0, 0))          # +roll torque (right side down in FLU)
    assert f[2] > f[0] and f[3] > f[1]          # left motors push harder
    f = run_mix(P, 0.5, (0, 0.05, 0))          # +pitch torque = nose down
    assert f[0] > f[1] and f[2] > f[3]          # rear motors push harder
    f = run_mix(P, 0.5, (0, 0, 0.02))          # +yaw torque (nose left)
    assert f[0] > f[1] and f[3] > f[2]          # CW props (M1, M4) push harder


def test_airmode_keeps_authority_at_zero_throttle(P):
    fmin = P[D.P_IDLE] ** 2
    f = run_mix(P, 0.0, (0.3, 0, 0), airmode=True)
    assert f.min() == pytest.approx(fmin)
    assert torques(P, f)[0] == pytest.approx(0.3, rel=1e-6)
    f2 = run_mix(P, 0.0, (0.3, 0, 0), airmode=False)
    assert torques(P, f2)[0] < 0.3 * 0.9          # without airmode it is clipped


def test_saturation_scales_and_bounds(P):
    f = run_mix(P, 0.9, (50.0, -30.0, 5.0))
    assert np.all(f >= P[D.P_IDLE] ** 2 - 1e-12) and np.all(f <= 1 + 1e-12)
    assert f.max() - f.min() == pytest.approx(1 - P[D.P_IDLE] ** 2)
    tq = torques(P, f)
    ratio = tq[0] / 50.0
    assert tq[1] / -30.0 == pytest.approx(ratio, rel=1e-6)    # direction preserved
