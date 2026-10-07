"""1 kHz quadrotor physics + Betaflight-like rate loop, compiled with numba.

Frames (see README "Frame conventions"):
  world  = the track-map frame: metres, z UP, gate yaw about +z.
  body   = FLU: x forward, y left, z up (thrust along +z_body).
  quaternion = [w, x, y, z], Hamilton, BODY -> WORLD  (v_w = q * v_b * q^-1).

Everything lives in flat float64 arrays so one numba call can advance many
ticks without Python overhead. Layouts are the S_*, C_*, P_* constants below.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                      # pragma: no cover - fallback is slow
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

# ---------------------------------------------------------------- state layout
S_P, S_V, S_Q, S_W, S_M = 0, 3, 6, 10, 13       # pos, vel(world), quat wxyz, omega(body), motor speed[0..1]
NS = 17
# ---------------------------------------------------------------- controller state
C_I, C_GF, C_DF, C_GPREV, C_SPS, C_SPPREV, C_SPLP, C_MC = 0, 3, 6, 9, 12, 15, 18, 21
NC = 25
# ---------------------------------------------------------------- params
_names = """DT G MASS IXX IYY IZZ ARM TMAX A_UP A_DN IDLE YAWK
DL0 DL1 DL2 DQ0 DQ1 DQ2 ROTD GE GEK RAD
KP0 KP1 KP2 KI0 KI1 KI2 KD0 KD1 KD2 KFF0 KFF1 KFF2 IL0 IL1 IL2
A_GYRO A_DTERM A_SPS A_RELAX RELAX_ON RELAX_THR AIRMODE GNOISE
FLOOR_CRASH GATE_CRASH REST FRIC BOUNCE GROUND_H CEIL_H CEIL_ON CEIL_CRASH
WALL_ON WALL_XMIN WALL_XMAX WALL_YMIN WALL_YMAX WALL_HALF WALL_H WALL_CRASH
MX0 MX1 MX2 MX3 MY0 MY1 MY2 MY3 MS0 MS1 MS2 MS3""".split()
PIDX = {n: i for i, n in enumerate(_names)}
NP = len(_names)
for _n, _i in PIDX.items():
    globals()["P_" + _n] = _i

# Log columns written by numba (the rest are filled by the app per batch).
LOG_COLUMNS = (
    ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz",
     "wx", "wy", "wz", "gyro_x", "gyro_y", "gyro_z",
     "sp_raw_x", "sp_raw_y", "sp_raw_z", "sp_x", "sp_y", "sp_z",
     "mcmd0", "mcmd1", "mcmd2", "mcmd3", "thrust0", "thrust1", "thrust2", "thrust3",
     "stick_roll", "stick_pitch", "stick_yaw", "stick_thr"]
    + [f"raw{i}" for i in range(8)]
    + ["next_station", "event_flags", "armed", "thr_cmd", "preset", "cam_tilt_deg"]
)
NLOG = len(LOG_COLUMNS)
COL = {n: i for i, n in enumerate(LOG_COLUMNS)}

EV_CRASH, EV_RESET, EV_ARMED_CHANGE, EV_LAP, EV_GATE, EV_LINE = 1, 2, 4, 8, 16, 32

# Gate collision boxes in GATE-LOCAL coords (+x normal, +y lateral, +z up,
# origin bottom-centre). Identical to the collision cubes of
# Gate collision boxes match the procedural frame rendered by render.py:
# 2.70 m outer, 1.50 m opening from z 0.60 to 2.10, 0.26 m deep.
GATE_BOXES = np.array([
    # cx,  cy,    cz,   hx,   hy,   hz
    [0.0, 0.00, 0.30, 0.13, 1.35, 0.30],   # bottom
    [0.0, 0.00, 2.40, 0.13, 1.35, 0.30],   # top
    [0.0, -1.05, 1.35, 0.13, 0.30, 0.75],  # left
    [0.0, 1.05, 1.35, 0.13, 0.30, 0.75],   # right
], dtype=np.float64)


def pt1_gain(cutoff_hz: float, dt: float) -> float:
    """Betaflight pt1FilterGain."""
    if cutoff_hz <= 0:
        return 1.0
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (rc + dt)


def build_params(cfg: dict) -> np.ndarray:
    ph, ct, co = cfg["physics"], cfg["controller"], cfg["collision"]
    dt = float(ph["dt"])
    P = np.zeros(NP)
    P[P_DT], P[P_G], P[P_MASS] = dt, ph["gravity"], ph["mass_kg"]
    P[P_IXX], P[P_IYY], P[P_IZZ] = ph["inertia_kgm2"]
    a = float(ph["arm_offset_m"])
    P[P_ARM], P[P_TMAX] = a, ph["motor_max_thrust_n"]
    P[P_A_UP] = 1.0 - math.exp(-dt / float(ph["motor_tau_up_s"]))
    P[P_A_DN] = 1.0 - math.exp(-dt / float(ph["motor_tau_down_s"]))
    P[P_IDLE], P[P_YAWK] = ph["motor_idle"], ph["yaw_torque_coef_m"]
    P[P_DL0:P_DL0 + 3] = ph["drag_linear"]
    P[P_DQ0:P_DQ0 + 3] = ph["drag_quadratic"]
    P[P_ROTD] = ph["rot_damping"]
    P[P_GE], P[P_GEK], P[P_RAD] = float(bool(ph["ground_effect"])), ph["ground_effect_gain"], ph["body_radius_m"]
    P[P_KP0:P_KP0 + 3] = ct["kp"]
    P[P_KI0:P_KI0 + 3] = ct["ki"]
    P[P_KD0:P_KD0 + 3] = ct["kd"]
    P[P_KFF0:P_KFF0 + 3] = ct["kff"]
    P[P_IL0:P_IL0 + 3] = ct["i_limit"]
    P[P_A_GYRO] = pt1_gain(ct["gyro_lpf_hz"], dt)
    P[P_A_DTERM] = pt1_gain(ct["dterm_lpf_hz"], dt)
    P[P_A_SPS] = pt1_gain(ct["setpoint_smoothing_hz"], dt)
    P[P_A_RELAX] = pt1_gain(ct["iterm_relax_cutoff_hz"], dt)
    P[P_RELAX_ON] = float(bool(ct["iterm_relax"]))
    P[P_RELAX_THR] = math.radians(float(ct["iterm_relax_threshold_dps"]))
    P[P_AIRMODE] = float(bool(ct["airmode"]))
    P[P_GNOISE] = math.radians(float(ct["gyro_noise_dps"]))
    P[P_FLOOR_CRASH], P[P_GATE_CRASH] = co["floor_crash_speed_mps"], co["gate_crash_speed_mps"]
    P[P_REST], P[P_FRIC] = co["restitution"], co["ground_friction"]
    P[P_BOUNCE] = 1.0 if co["mode"] == "bounce" else 0.0
    P[P_GROUND_H] = cfg["track"]["spawn_height_m"]
    ceil_h = cfg["track"].get("ceiling_height_m")
    P[P_CEIL_ON] = 0.0 if ceil_h is None else 1.0
    P[P_CEIL_H] = 0.0 if ceil_h is None else float(ceil_h)
    P[P_CEIL_CRASH] = 1.0 if cfg["track"].get("ceiling_crash") else 0.0
    P[P_WALL_CRASH] = co.get("wall_crash_speed_mps", co["gate_crash_speed_mps"])
    # Betaflight QUAD-X motor order: 1 rear-right, 2 front-right, 3 rear-left, 4 front-left.
    # Positions in body FLU (y left). props-in: M1 CW, M2 CCW, M3 CCW, M4 CW.
    # A CW prop (seen from above) pushes the frame CCW = +z torque.
    P[P_MX0:P_MX0 + 4] = [-a, a, -a, a]
    P[P_MY0:P_MY0 + 4] = [-a, -a, a, a]
    spin = [1.0, -1.0, -1.0, 1.0]
    if not ph.get("props_in", True):
        spin = [-s for s in spin]
    P[P_MS0:P_MS0 + 4] = spin
    return P


def configure_venue_bounds(P: np.ndarray, bbox: tuple, cfg: dict) -> None:
    """Match physical wall/roof boxes to the venue geometry drawn around a track."""
    if not cfg["track"].get("venue_walls", True):
        P[P_WALL_ON] = 0.0
        return
    margin = float(cfg["track"].get("venue_margin_m", 12.0))
    half_thickness = float(cfg["track"].get("venue_wall_thickness_m", 0.4)) / 2.0
    xmin, xmax, ymin, ymax = bbox
    P[P_WALL_ON] = 1.0
    P[P_WALL_XMIN] = xmin - margin
    P[P_WALL_XMAX] = xmax + margin
    P[P_WALL_YMIN] = ymin - margin
    P[P_WALL_YMAX] = ymax + margin
    P[P_WALL_HALF] = half_thickness
    P[P_WALL_H] = (P[P_CEIL_H] if P[P_CEIL_ON] > 0
                   else float(cfg["track"].get("venue_wall_height_m", 8.0)))


@njit(cache=True)
def quat_to_R(q, R):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R[0, 0] = 1 - 2 * (y * y + z * z); R[0, 1] = 2 * (x * y - w * z); R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z); R[1, 1] = 1 - 2 * (x * x + z * z); R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y); R[2, 1] = 2 * (y * z + w * x); R[2, 2] = 1 - 2 * (x * x + y * y)


@njit(cache=True)
def mix(P, thr, tx, ty, tz, airmode, out_f):
    """Torque demand (N m, body FLU) + throttle stick [0,1] -> per-motor thrust fractions.

    Throttle maps to motor COMMAND like Betaflight (c = idle + (1-idle)*thr), and
    collective thrust fraction is c^2. Attitude corrections are allocated in
    thrust space (exact inverse of the rigid-body torque map), then Betaflight-
    style: if the correction range exceeds the available range it is scaled
    down; with airmode the whole mix is shifted to stay inside [idle^2, 1].
    """
    idle = P[P_IDLE]
    tmax = P[P_TMAX]
    a = P[P_ARM]
    c_thr = idle + (1.0 - idle) * thr
    f_thr = c_thr * c_thr
    fmin = idle * idle
    lo = 1e9
    hi = -1e9
    for i in range(4):
        d = (tx * P[P_MY0 + i] / (4 * a * a) - ty * P[P_MX0 + i] / (4 * a * a)
             + tz * P[P_MS0 + i] / (4 * P[P_YAWK])) / tmax
        out_f[i] = d
        if d < lo:
            lo = d
        if d > hi:
            hi = d
    avail = 1.0 - fmin
    rng = hi - lo
    if rng > avail:
        s = avail / rng
        for i in range(4):
            out_f[i] *= s
        lo *= s
        hi *= s
    shift = 0.0
    if airmode:
        if f_thr + hi > 1.0:
            shift = 1.0 - (f_thr + hi)
        elif f_thr + lo < fmin:
            shift = fmin - (f_thr + lo)
    for i in range(4):
        f = f_thr + out_f[i] + shift
        if f < fmin:
            f = fmin
        if f > 1.0:
            f = 1.0
        out_f[i] = f


@njit(cache=True)
def init_crossings(s, E, dprev):
    for j in range(E.shape[0]):
        dprev[j] = (s[S_P] - E[j, 0]) * E[j, 3] + (s[S_P + 1] - E[j, 1]) * E[j, 4]


@njit(cache=True)
def _sphere_box_contact(s, radius, cx, cy, cz, hx, hy, hz):
    """Sphere/AABB contact as (hit, normal xyz, penetration)."""
    ex, ey, ez = s[S_P] - cx, s[S_P + 1] - cy, s[S_P + 2] - cz
    qx = min(max(ex, -hx), hx)
    qy = min(max(ey, -hy), hy)
    qz = min(max(ez, -hz), hz)
    dx, dy, dz = ex - qx, ey - qy, ez - qz
    distance_sq = dx * dx + dy * dy + dz * dz
    if distance_sq >= radius * radius:
        return False, 0.0, 0.0, 0.0, 0.0
    if distance_sq > 1e-18:
        distance = math.sqrt(distance_sq)
        return True, dx / distance, dy / distance, dz / distance, radius - distance

    # The sphere centre is inside the box. Use the nearest face as the exit.
    gx, gy, gz = hx - abs(ex), hy - abs(ey), hz - abs(ez)
    if gx <= gy and gx <= gz:
        return True, (1.0 if ex >= 0.0 else -1.0), 0.0, 0.0, radius + gx
    if gy <= gz:
        return True, 0.0, (1.0 if ey >= 0.0 else -1.0), 0.0, radius + gy
    return True, 0.0, 0.0, (1.0 if ez >= 0.0 else -1.0), radius + gz


@njit(cache=True)
def _tick(s, c, P, G, sp, thr, armed, R, fbuf):
    """Advance one physics tick. Returns 0 ok or a surface-specific crash code."""
    dt = P[P_DT]
    # ---------------- simulated gyro + rate PID
    alpha = np.empty(3)
    for i in range(3):
        g = s[S_W + i]
        if P[P_GNOISE] > 0:
            g += P[P_GNOISE] * np.random.normal()
        gf = c[C_GF + i] + P[P_A_GYRO] * (g - c[C_GF + i])
        c[C_GF + i] = gf
        sps = c[C_SPS + i] + P[P_A_SPS] * (sp[i] - c[C_SPS + i])
        c[C_SPS + i] = sps
        dsp = (sps - c[C_SPPREV + i]) / dt
        c[C_SPPREV + i] = sps
        err = sps - gf
        dg = (gf - c[C_GPREV + i]) / dt
        c[C_GPREV + i] = gf
        df = c[C_DF + i] + P[P_A_DTERM] * (dg - c[C_DF + i])
        c[C_DF + i] = df
        lp = c[C_SPLP + i] + P[P_A_RELAX] * (sps - c[C_SPLP + i])
        c[C_SPLP + i] = lp
        relax = 1.0
        if P[P_RELAX_ON] > 0:
            relax = 1.0 - abs(sps - lp) / P[P_RELAX_THR]
            if relax < 0.0:
                relax = 0.0
        if armed:
            it = c[C_I + i] + P[P_KI0 + i] * err * dt * relax
            lim = P[P_IL0 + i]
            if it > lim:
                it = lim
            if it < -lim:
                it = -lim
            c[C_I + i] = it
        else:
            c[C_I + i] = 0.0
        alpha[i] = P[P_KP0 + i] * err + c[C_I + i] - P[P_KD0 + i] * df + P[P_KFF0 + i] * dsp
    wx, wy, wz = s[S_W], s[S_W + 1], s[S_W + 2]
    Ix, Iy, Iz = P[P_IXX], P[P_IYY], P[P_IZZ]
    # torque = J*alpha + w x Jw (gyroscopic compensation)
    tx = Ix * alpha[0] + (wy * Iz * wz - wz * Iy * wy)
    ty = Iy * alpha[1] + (wz * Ix * wx - wx * Iz * wz)
    tz = Iz * alpha[2] + (wx * Iy * wy - wy * Ix * wx)
    # ---------------- mixer + motors
    if armed:
        mix(P, thr, tx, ty, tz, P[P_AIRMODE] > 0, fbuf)
    T = 0.0
    mtx = 0.0
    mty = 0.0
    mtz = 0.0
    ge = 1.0
    if P[P_GE] > 0:
        ge = 1.0 + P[P_GEK] * math.exp(-max(s[S_P + 2], 0.0) / 0.15)
    for i in range(4):
        cmd = math.sqrt(fbuf[i]) if armed else 0.0
        c[C_MC + i] = cmd
        m = s[S_M + i]
        if cmd > m:
            m += P[P_A_UP] * (cmd - m)
        else:
            m += P[P_A_DN] * (cmd - m)
        s[S_M + i] = m
        Ti = P[P_TMAX] * m * m * ge
        T += Ti
        mtx += P[P_MY0 + i] * Ti
        mty -= P[P_MX0 + i] * Ti
        mtz += P[P_YAWK] * P[P_MS0 + i] * Ti
    # ---------------- rigid body
    quat_to_R(s[S_Q:S_Q + 4], R)
    vx, vy, vz = s[S_V], s[S_V + 1], s[S_V + 2]
    vb0 = R[0, 0] * vx + R[1, 0] * vy + R[2, 0] * vz
    vb1 = R[0, 1] * vx + R[1, 1] * vy + R[2, 1] * vz
    vb2 = R[0, 2] * vx + R[1, 2] * vy + R[2, 2] * vz
    fb0 = -(P[P_DL0] * vb0 + P[P_DQ0] * abs(vb0) * vb0)
    fb1 = -(P[P_DL1] * vb1 + P[P_DQ1] * abs(vb1) * vb1)
    fb2 = T - (P[P_DL2] * vb2 + P[P_DQ2] * abs(vb2) * vb2)
    m_ = P[P_MASS]
    ax = (R[0, 0] * fb0 + R[0, 1] * fb1 + R[0, 2] * fb2) / m_
    ay = (R[1, 0] * fb0 + R[1, 1] * fb1 + R[1, 2] * fb2) / m_
    az = (R[2, 0] * fb0 + R[2, 1] * fb1 + R[2, 2] * fb2) / m_ - P[P_G]
    rd = P[P_ROTD]
    dwx = (mtx - rd * wx - (wy * Iz * wz - wz * Iy * wy)) / Ix
    dwy = (mty - rd * wy - (wz * Ix * wx - wx * Iz * wz)) / Iy
    dwz = (mtz - rd * wz - (wx * Iy * wy - wy * Ix * wx)) / Iz
    # semi-implicit Euler + exact quaternion exponential for the rotation
    s[S_V] += ax * dt; s[S_V + 1] += ay * dt; s[S_V + 2] += az * dt
    s[S_P] += s[S_V] * dt; s[S_P + 1] += s[S_V + 1] * dt; s[S_P + 2] += s[S_V + 2] * dt
    s[S_W] += dwx * dt; s[S_W + 1] += dwy * dt; s[S_W + 2] += dwz * dt
    wx, wy, wz = s[S_W], s[S_W + 1], s[S_W + 2]
    wn = math.sqrt(wx * wx + wy * wy + wz * wz)
    if wn > 1e-12:
        h = 0.5 * wn * dt
        k = math.sin(h) / wn
        dw, dx, dy, dz = math.cos(h), wx * k, wy * k, wz * k
        qw, qx, qy, qz = s[S_Q], s[S_Q + 1], s[S_Q + 2], s[S_Q + 3]
        nw = qw * dw - qx * dx - qy * dy - qz * dz
        nx = qw * dx + qx * dw + qy * dz - qz * dy
        ny = qw * dy - qx * dz + qy * dw + qz * dx
        nz = qw * dz + qx * dy - qy * dx + qz * dw
        nn = math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        s[S_Q] = nw / nn; s[S_Q + 1] = nx / nn; s[S_Q + 2] = ny / nn; s[S_Q + 3] = nz / nn
    # ---------------- floor
    bounce = P[P_BOUNCE] > 0
    h0 = P[P_GROUND_H]
    if s[S_P + 2] < h0:
        impact = -s[S_V + 2]
        if (not bounce) and impact > P[P_FLOOR_CRASH]:
            return 1
        s[S_P + 2] = h0
        if s[S_V + 2] < 0:
            s[S_V + 2] = -s[S_V + 2] * P[P_REST] if impact > 1.0 else 0.0
        f = 1.0 - P[P_FRIC] * dt
        if f < 0:
            f = 0.0
        s[S_V] *= f
        s[S_V + 1] *= f
        fw = 1.0 - 20.0 * dt
        s[S_W] *= fw; s[S_W + 1] *= fw; s[S_W + 2] *= fw
    # ---------------- finite venue walls and roof
    if P[P_WALL_ON] > 0:
        r = P[P_RAD]
        xmin, xmax = P[P_WALL_XMIN], P[P_WALL_XMAX]
        ymin, ymax = P[P_WALL_YMIN], P[P_WALL_YMAX]
        half, height = P[P_WALL_HALF], P[P_WALL_H]
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        hx, hy = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
        box_count = 5 if P[P_CEIL_ON] > 0 else 4
        for box_index in range(box_count):
            if box_index == 0:
                bx, by, bz, bhx, bhy, bhz = cx, ymin, height / 2.0, hx, half, height / 2.0
            elif box_index == 1:
                bx, by, bz, bhx, bhy, bhz = cx, ymax, height / 2.0, hx, half, height / 2.0
            elif box_index == 2:
                bx, by, bz, bhx, bhy, bhz = xmin, cy, height / 2.0, half, hy, height / 2.0
            elif box_index == 3:
                bx, by, bz, bhx, bhy, bhz = xmax, cy, height / 2.0, half, hy, height / 2.0
            else:
                bx, by, bz, bhx, bhy, bhz = cx, cy, height + half, hx, hy, half
            hit, nx, ny, nz, penetration = _sphere_box_contact(
                s, r, bx, by, bz, bhx, bhy, bhz)
            if not hit:
                continue
            vn = s[S_V] * nx + s[S_V + 1] * ny + s[S_V + 2] * nz
            impact = max(-vn, 0.0)
            roof_hit = box_index == 4
            hard_hit = (P[P_CEIL_CRASH] > 0 and impact > P[P_FLOOR_CRASH]
                        if roof_hit else impact > P[P_WALL_CRASH])
            # Resolve overlap even on a crash tick so retained state never sits
            # inside or beyond the visible geometry.
            s[S_P] += nx * penetration
            s[S_P + 1] += ny * penetration
            s[S_P + 2] += nz * penetration
            if (not bounce) and hard_hit:
                return 3 if roof_hit else 4
            if vn < 0.0:
                k2 = (1.0 + P[P_REST]) * vn
                s[S_V] -= k2 * nx
                s[S_V + 1] -= k2 * ny
                s[S_V + 2] -= k2 * nz
    # ---------------- gate frames
    r = P[P_RAD]
    for j in range(G.shape[0]):
        dx = s[S_P] - G[j, 0]
        dy = s[S_P + 1] - G[j, 1]
        lz = s[S_P + 2] - G[j, 2]
        co, si = G[j, 3], G[j, 4]
        lx = dx * co + dy * si
        if abs(lx) > 0.13 + r:
            continue
        ly = -dx * si + dy * co
        if abs(ly) > 1.35 + r or lz < -r or lz > 2.70 + r:
            continue
        for b in range(4):
            ex = lx - GATE_BOXES[b, 0]; ey = ly - GATE_BOXES[b, 1]; ez = lz - GATE_BOXES[b, 2]
            qx_ = min(max(ex, -GATE_BOXES[b, 3]), GATE_BOXES[b, 3])
            qy_ = min(max(ey, -GATE_BOXES[b, 4]), GATE_BOXES[b, 4])
            qz_ = min(max(ez, -GATE_BOXES[b, 5]), GATE_BOXES[b, 5])
            ddx, ddy, ddz = ex - qx_, ey - qy_, ez - qz_
            dist = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            if dist >= r:
                continue
            if dist < 1e-9:                      # centre inside the bar
                ddx, ddy, ddz, dist = (1.0 if ex >= 0 else -1.0), 0.0, 0.0, 0.0
                nlx, nly, nlz = ddx, 0.0, 0.0
            else:
                nlx, nly, nlz = ddx / dist, ddy / dist, ddz / dist
            nwx = nlx * co - nly * si
            nwy = nlx * si + nly * co
            nwz = nlz
            vn = s[S_V] * nwx + s[S_V + 1] * nwy + s[S_V + 2] * nwz
            if (not bounce) and -vn > P[P_GATE_CRASH]:
                return 2
            pen = r - dist
            s[S_P] += nwx * pen; s[S_P + 1] += nwy * pen; s[S_P + 2] += nwz * pen
            if vn < 0:
                k2 = (1.0 + P[P_REST]) * vn
                s[S_V] -= k2 * nwx; s[S_V + 1] -= k2 * nwy; s[S_V + 2] -= k2 * nwz
    return 0


@njit(cache=True)
def step_n(s, c, P, G, E, dprev, sp, thr, armed, n, t0, L, T, row0, xb):
    """Advance up to n ticks with constant stick input.

    L/T: log buffers (rows row0..row0+n-1 are written; pass row0 < 0 to skip).
    xb:  crossing buffer rows [element, t, lateral, z, direction].
    Returns (ticks_done, event_code, n_crossings). Stops at a crash tick.
    Event codes: 1 floor, 2 gate, 3 ceiling, 4 venue wall.
    """
    R = np.empty((3, 3))
    fbuf = np.zeros(4)
    for i in range(4):
        fbuf[i] = P[P_IDLE] * P[P_IDLE]
    dt = P[P_DT]
    ncross = 0
    for k in range(n):
        px, py, pz = s[S_P], s[S_P + 1], s[S_P + 2]
        ev = _tick(s, c, P, G, sp, thr, armed, R, fbuf)
        t = t0 + (k + 1) * dt
        # plane crossings (inside-aperture test happens here, per tick)
        for j in range(E.shape[0]):
            d = (s[S_P] - E[j, 0]) * E[j, 3] + (s[S_P + 1] - E[j, 1]) * E[j, 4]
            dp = dprev[j]
            dprev[j] = d
            if (dp < 0.0 and d >= 0.0) or (dp > 0.0 and d <= 0.0):
                f = dp / (dp - d)
                cx = px + f * (s[S_P] - px)
                cy = py + f * (s[S_P + 1] - py)
                cz = pz + f * (s[S_P + 2] - pz)
                lat = -(cx - E[j, 0]) * E[j, 4] + (cy - E[j, 1]) * E[j, 3]
                if abs(lat) <= E[j, 5] and cz >= E[j, 6] and cz <= E[j, 7] and ncross < xb.shape[0]:
                    xb[ncross, 0] = j
                    xb[ncross, 1] = t - dt + f * dt
                    xb[ncross, 2] = lat
                    xb[ncross, 3] = cz
                    xb[ncross, 4] = 1.0 if d > dp else -1.0
                    ncross += 1
        if row0 >= 0:
            rr = row0 + k
            T[rr] = t
            for i in range(13):
                L[rr, i] = s[i]
            for i in range(3):
                L[rr, 13 + i] = c[C_GF + i]
                L[rr, 16 + i] = sp[i]
                L[rr, 19 + i] = c[C_SPS + i]
            for i in range(4):
                L[rr, 22 + i] = c[C_MC + i]
                L[rr, 26 + i] = P[P_TMAX] * s[S_M + i] * s[S_M + i]
        if ev != 0:
            return k + 1, ev, ncross
    return n, 0, ncross


def new_state(pos, yaw):
    s = np.zeros(NS)
    s[S_P:S_P + 3] = pos
    s[S_Q] = math.cos(yaw / 2)
    s[S_Q + 3] = math.sin(yaw / 2)
    return s


def new_ctrl():
    return np.zeros(NC)


def hover_throttle_stick(P) -> float:
    """Throttle stick value whose collective thrust equals weight (static)."""
    f = P[P_MASS] * P[P_G] / (4 * P[P_TMAX])
    c = math.sqrt(f)
    return (c - P[P_IDLE]) / (1 - P[P_IDLE])


def quat_to_R_np(q):
    R = np.empty((3, 3))
    quat_to_R(np.asarray(q, float), R)
    return R
