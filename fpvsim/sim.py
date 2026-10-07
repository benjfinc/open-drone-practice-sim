"""Simulation core (no window): physics batches, arming, crash/reset, laps, logging."""
from __future__ import annotations

import math

import numpy as np

from . import dynamics as D
from . import laps as lapsmod
from . import track_env
from .dynamics import COL, EV_ARMED_CHANGE, EV_CRASH, EV_GATE, EV_LAP, EV_LINE, EV_RESET
from .rates import RateProfile, stick_to_body_rates


class Sim:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.env = track_env.load(cfg)
        self.P = D.build_params(cfg)
        D.configure_venue_bounds(self.P, self.env.bbox, cfg)
        self.dt = float(self.P[D.P_DT])
        r = cfg["rates"]
        self.profiles = {name: RateProfile(name, p, r["rc_deadband"], r["yaw_deadband"],
                                           r["throttle"]["mid"], r["throttle"]["expo"])
                         for name, p in r["presets"].items()}
        self.preset_names = list(self.profiles)
        self.preset_cycle = [p for p in r.get("preset_cycle", self.preset_names) if p in self.profiles]
        self.preset = r["default_preset"]
        self.laps = lapsmod.LapTracker(self.env, bool(cfg["track"].get("require_forward_direction", True)))
        self.logger = logger
        self._L = np.zeros((1, D.NLOG), np.float32)
        self._T = np.zeros(1)
        self.xb = np.zeros((64, 5))
        self.dprev = np.zeros(len(self.env.E))
        self.tick = 0
        self.armed = False
        self.crashed_at = None
        self.crash_info = None
        self.tilt_deg = float(cfg["camera"]["uptilt_deg"])
        self.aborted, self.n_crashes, self.n_resets = [], 0, 0
        self.flags = 0
        self.messages = []
        self._disarmed_throttle_prompted = False
        self._high_throttle_arm_prompted = False
        self.last_thr, self.last_sp_dps = 0.0, (0.0, 0.0, 0.0)
        self.elements = {e.eid: e for e in self.env.track.elements}
        self.crash_delay = float(cfg["collision"]["crash_reset_delay_s"])
        self.s = D.new_state(self.env.spawn_pos, self.env.spawn_yaw)
        self.c = D.new_ctrl()
        self.reset("start")
        # JIT warm-up (compiles or loads the numba cache) on scratch copies
        D.step_n(self.s.copy(), self.c.copy(), self.P, self.env.G, self.env.E, self.dprev.copy(),
                 np.zeros(3), 0.0, False, 1, 0.0, self._L, self._T, -1, self.xb)

    # ------------------------------------------------------------------ helpers
    @property
    def t(self):
        return self.tick * self.dt

    @property
    def profile(self):
        return self.profiles[self.preset]

    def msg(self, text, dur=2.0):
        self.messages.append((self.t + dur, text))

    def active_messages(self):
        self.messages = [(te, m) for te, m in self.messages if te > self.t]
        return [m for _, m in self.messages[-3:]]

    def _event(self, kind, **kw):
        if self.logger:
            kw.setdefault("t", self.t)
            self.logger.event(kind, **kw)

    def cycle_preset(self, step=1):
        cyc = self.preset_cycle
        i = cyc.index(self.preset) if self.preset in cyc else -1
        self.set_preset(cyc[(i + step) % len(cyc)])

    def set_preset(self, name):
        if name != self.preset and name in self.profiles:
            self.preset = name
            self._event("preset", name=name, max_dps=self.profile.max_rates_dps())
            self.msg(f"RATES {name.upper()}")

    def set_tilt(self, deg):
        cc = self.cfg["camera"]
        self.tilt_deg = float(min(max(deg, cc["uptilt_min_deg"]), cc["uptilt_max_deg"]))
        self._event("tilt", deg=self.tilt_deg)

    def next_target(self):
        eid = self.laps.next_checkpoint
        e = self.elements[eid]
        return eid, (e.pos[0], e.pos[1], e.pos[2] + 1.35)

    # ------------------------------------------------------------------ state changes
    def reset(self, reason):
        e = self.env
        self.s[:] = D.new_state(e.spawn_pos, e.spawn_yaw)
        self.c[:] = D.new_ctrl()
        D.init_crossings(self.s, e.E, self.dprev)
        info = self.laps.abort(self.t, reason)
        if info:
            self.aborted.append(info)
        self.crashed_at = None
        if reason != "start":
            self.n_resets += 1
        self.flags |= EV_RESET
        self._event("reset", reason=reason, aborted_lap=info)

    def set_armed(self, want, throttle_stick, arm_hint="PRESS SPACE"):
        if want and not self.armed:
            if throttle_stick > 0.05:
                if not self._high_throttle_arm_prompted:
                    self.msg("THROTTLE LOW TO ARM", 2.0)
                    self._high_throttle_arm_prompted = True
                return
            self._high_throttle_arm_prompted = False
            self._disarmed_throttle_prompted = False
            self.armed = True
            self.flags |= EV_ARMED_CHANGE
            self._event("arm")
        elif not want and self.armed:
            self._high_throttle_arm_prompted = False
            self._disarmed_throttle_prompted = False
            self.armed = False
            self.flags |= EV_ARMED_CHANGE
            self._event("disarm")
        elif not want:
            self._high_throttle_arm_prompted = False
            throttle_raised = throttle_stick > 0.05
            if throttle_raised and not self._disarmed_throttle_prompted:
                self.msg(f"ARM FIRST — {arm_hint}", 2.5)
            self._disarmed_throttle_prompted = throttle_raised

    # ------------------------------------------------------------------ stepping
    def advance(self, n, sticks):
        prof = self.profile
        sp_dps = prof.setpoints_dps(sticks.roll, sticks.pitch, sticks.yaw)
        sp = np.array(stick_to_body_rates(sp_dps))
        thr = prof.throttle(sticks.throttle)
        self.last_sp_dps, self.last_thr = sp_dps, thr
        if self.crashed_at is not None:
            self.tick += n
            if self.t - self.crashed_at >= self.crash_delay:
                self.reset("crash")
            return
        lg = self.logger
        while n > 0:
            m = min(n, lg.cap) if lg else n
            row = lg.reserve(m) if lg else -1
            L, T = (lg.L, lg.T) if lg else (self._L, self._T)
            t0 = self.t
            done, ev, nx = D.step_n(self.s, self.c, self.P, self.env.G, self.env.E, self.dprev,
                                    sp, thr, self.armed, m, t0, L, T, row, self.xb)
            self.tick += done
            flag_rows = {0: self.flags}
            self.flags = 0
            for i in range(nx):
                j, tc, lat, z, dirn = self.xb[i]
                k = int(min(max((tc - t0) / self.dt, 0), done - 1))
                fl = EV_LINE if int(j) == 0 else 0
                for kind, data in self.laps.on_crossing(int(j), float(tc), float(lat), float(z), float(dirn)):
                    self._event(kind, **data)
                    if kind == "gate":
                        fl |= EV_GATE
                        self.msg(f"{data['checkpoint']}  {data['split_s']:.2f}")
                    elif kind == "lap":
                        fl |= EV_LAP
                        txt = f"LAP {data['time_s']:.2f}"
                        if data["missed"]:
                            txt += "  MISSED " + ",".join(data["missed"])
                        self.msg(txt, 4.0)
                    elif kind == "gate_missed":
                        self.msg(f"MISSED {data['checkpoint']}")
                flag_rows[k] = flag_rows.get(k, 0) | fl
            if ev:
                self.n_crashes += 1
                self.crashed_at = self.t
                speed = float(np.linalg.norm(self.s[3:6]))
                self.crash_info = {"hit": {1: "floor", 2: "gate", 3: "ceiling", 4: "wall"}.get(ev, "?"), "speed_mps": speed,
                                   "pos": [float(x) for x in self.s[0:3]]}
                self._event("crash", **self.crash_info)
                flag_rows[done - 1] = flag_rows.get(done - 1, 0) | EV_CRASH
                self.msg(f"CRASH ({self.crash_info['hit']}, {speed:.1f} m/s)", self.crash_delay)
            if lg:
                sl = slice(row, row + done)
                L[sl, COL["stick_roll"]] = sticks.roll
                L[sl, COL["stick_pitch"]] = sticks.pitch
                L[sl, COL["stick_yaw"]] = sticks.yaw
                L[sl, COL["stick_thr"]] = sticks.throttle
                L[sl, COL["raw0"]:COL["raw0"] + 8] = np.asarray(sticks.raw[:8], np.float32)
                L[sl, COL["next_station"]] = self.laps.next_idx if self.laps.started else -1
                L[sl, COL["event_flags"]] = 0
                for k, fl in flag_rows.items():
                    if fl:
                        L[row + k, COL["event_flags"]] = float(fl)
                L[sl, COL["armed"]] = 1.0 if self.armed else 0.0
                L[sl, COL["thr_cmd"]] = thr
                L[sl, COL["preset"]] = self.preset_names.index(self.preset)
                L[sl, COL["cam_tilt_deg"]] = self.tilt_deg
                lg.commit(done)
            n -= done
            if ev:
                self.tick += n                  # time still passes during the crash pause
                break

    def close(self):
        if self.logger:
            self.logger.close(self.laps.laps, self.aborted, self.laps.crossings)
