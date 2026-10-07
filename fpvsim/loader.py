"""Load recorded runs and derive camera poses for re-rendering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .dynamics import EV_RESET, EV_CRASH, quat_to_R_np


class Run:
    def __init__(self, path):
        self.dir = Path(path)
        d = self.dir
        if (d / "ticks.npz").exists():
            z = np.load(d / "ticks.npz")
            self.t, self.data, self.columns = z["t"], z["data"], [str(c) for c in z["columns"]]
        else:                                           # run was killed: read the chunks
            ch = d / "chunks"
            tk = sorted(ch.glob("ticks_*.npy"))
            self.data = np.concatenate([np.load(p) for p in tk]) if tk else np.zeros((0, 0), np.float32)
            tt = sorted(ch.glob("t_*.npy"))
            self.t = np.concatenate([np.load(p) for p in tt]) if tt else np.zeros(0)
            self.columns = json.loads((d / "meta.json").read_text())["tick_columns"]
        self.col = {c: i for i, c in enumerate(self.columns)}
        self.frames = self.frame_columns = None
        if (d / "frames.npz").exists():
            z = np.load(d / "frames.npz")
            self.frames, self.frame_columns = z["data"], [str(c) for c in z["columns"]]
        self.events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()] \
            if (d / "events.jsonl").exists() else []
        self.laps = json.loads((d / "laps.json").read_text()) if (d / "laps.json").exists() else None
        self.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        self.config = yaml.safe_load((d / "config_used.yaml").read_text()) \
            if (d / "config_used.yaml").exists() else None

    def __getitem__(self, name):
        return self.data[:, self.col[name]]

    def get(self, *names):
        return np.stack([self[n] for n in names], -1)

    @property
    def pos(self):
        return self.get("px", "py", "pz")

    @property
    def quat_wxyz(self):
        return self.get("qw", "qx", "qy", "qz")

    def segments(self):
        """Segment id per tick; increments at every reset/crash (pose jumps there)."""
        fl = self["event_flags"].astype(np.int64)
        return np.cumsum((fl & (EV_RESET | EV_CRASH)) != 0)


def load_run(path) -> Run:
    return Run(path)


def camera_poses(run: Run, rate_hz=30.0, tilt_deg=None, offset_body=None, armed_only=True):
    """Resample the 1 kHz log at a camera rate and return FPV camera poses.

    Returns dict of arrays: t, cam_pos (N,3), cam_quat_wxyz (N,4) in the OpenGL /
    USD camera convention (+X right, +Y up, -Z forward, camera->world) -- the same
    OpenGL/USD convention -- plus body pose,
    tilt used and segment id. tilt_deg=None uses the tilt logged at each tick.
    """
    from .render import fpv_camera_R_body
    t = run.t
    if len(t) == 0:
        return None
    times = np.arange(t[0], t[-1], 1.0 / rate_hz)
    idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    if armed_only:
        idx = idx[run["armed"][idx] > 0.5]
    off = np.asarray(offset_body if offset_body is not None
                     else (run.config or {}).get("camera", {}).get("offset_body_m", [0.05, 0, 0.02]), float)
    P, Q = run.pos[idx].astype(float), run.quat_wxyz[idx].astype(float)
    tilts = run["cam_tilt_deg"][idx] if tilt_deg is None else np.full(len(idx), float(tilt_deg))
    cam_p, cam_q = np.zeros((len(idx), 3)), np.zeros((len(idx), 4))
    from .render import mat_to_quat
    for k in range(len(idx)):
        R_wb = quat_to_R_np(Q[k])
        cam_p[k] = P[k] + R_wb @ off
        cam_q[k] = mat_to_quat(R_wb @ fpv_camera_R_body(tilts[k]))
    return {"t": t[idx], "tick_index": idx, "cam_pos": cam_p, "cam_quat_wxyz": cam_q,
            "body_pos": P, "body_quat_wxyz": Q, "tilt_deg": tilts, "segment": run.segments()[idx]}
