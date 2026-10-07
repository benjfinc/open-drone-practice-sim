"""Convert the YAML course into arrays used by physics and lap scoring."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import REPO  # noqa: E402
from . import track  # noqa: E402

OPEN_HALF_W = 0.75
OPEN_Z_LO, OPEN_Z_HI = 0.60, 2.10
TIMING_LINE_HALF_LEN = 1.173


def station_of(eid: str) -> str:
    return eid[:-1] if eid.endswith("b") else eid


@dataclass
class TrackEnv:
    track: object
    element_ids: list          # index into E; 0 = timing line
    G: np.ndarray              # gates: cx, cy, cz(bottom), cos(yaw), sin(yaw)
    gate_ids: list
    E: np.ndarray              # crossing elements: cx,cy,cz,nx,ny,half_w,z_lo,z_hi
    sequence: list             # required checkpoint ids per lap, in order
    mode: str
    spawn_pos: tuple
    spawn_yaw: float
    bbox: tuple                # xmin, xmax, ymin, ymax of all elements

    def checkpoint_of(self, eid: str):
        """Which checkpoint in `sequence` a crossed element satisfies (or None)."""
        if eid in self.sequence:
            return eid
        st = station_of(eid)
        return st if st in self.sequence else None


def load(cfg: dict) -> TrackEnv:
    tc = cfg["track"]
    path = Path(tc["yaml"])
    if not path.is_absolute():
        path = REPO / path
    T = track.load(path)
    gates = T.gates
    tl = T.timing_line
    assert not tl.is_gate and tl.eid not in [g.eid for g in gates]

    G = np.array([[g.pos[0], g.pos[1], g.pos[2], math.cos(g.yaw), math.sin(g.yaw)] for g in gates])
    margin = float(tc.get("gate_pass_margin_m", 0.0))
    rows = [[tl.pos[0], tl.pos[1], tl.pos[2], math.cos(tl.yaw), math.sin(tl.yaw),
             TIMING_LINE_HALF_LEN, -1.0, 6.0]]
    for g in gates:
        rows.append([g.pos[0], g.pos[1], g.pos[2], math.cos(g.yaw), math.sin(g.yaw),
                     OPEN_HALF_W + margin, g.pos[2] + OPEN_Z_LO - margin, g.pos[2] + OPEN_Z_HI + margin])
    E = np.array(rows)

    mode = tc.get("double_gate_mode", "one_station")
    order = [str(x) for x in T.lap_order]
    if order and order[0] == tl.eid:
        order = order[1:]
    if mode == "one_station":
        seq = [station_of(e) for e in order]
    elif mode == "sequential":
        seq = []
        dg = [str(x) for x in tc.get("double_gate_sequence", [])]
        for e in order:
            st = station_of(e)
            if dg and st == station_of(dg[0]):
                seq.extend(dg)
            else:
                seq.append(e)
    else:
        raise ValueError(f"double_gate_mode {mode!r}")
    gate_eids = {g.eid for g in gates} | {station_of(g.eid) for g in gates}
    for s in seq:
        assert s in gate_eids, f"lap_order entry {s} is not a gate"

    back = float(tc.get("spawn_back_m", 2.0))
    spawn = (tl.pos[0] - back * math.cos(tl.yaw), tl.pos[1] - back * math.sin(tl.yaw),
             float(tc.get("spawn_height_m", 0.03)))
    xs = [e.pos[0] for e in T.elements]
    ys = [e.pos[1] for e in T.elements]
    return TrackEnv(T, [tl.eid] + [g.eid for g in gates], G, [g.eid for g in gates], E, seq, mode,
                    spawn, tl.yaw, (min(xs), max(xs), min(ys), max(ys)))
