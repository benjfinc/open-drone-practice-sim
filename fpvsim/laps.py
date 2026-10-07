"""Lap timing from in-aperture plane crossings.

Rules:
- A lap starts at a forward crossing of the timing line.
- The next forward timing-line crossing ends the lap. If no checkpoint was
  passed yet, it just restarts the timer (hovering over the line is not a lap).
- Checkpoints are expected in `sequence` order. Passing a later one marks the
  skipped ones MISSED and continues from there; passing an earlier/repeated one
  is logged but ignored. A lap is VALID only if nothing was missed.
- Backward crossings (direction -1) never count when require_forward is set.
"""
from __future__ import annotations


class LapTracker:
    def __init__(self, env, require_forward: bool = True):
        self.env = env
        self.seq = list(env.sequence)
        self.require_forward = require_forward
        self.laps = []          # completed laps
        self.crossings = []     # every in-aperture crossing, raw
        self.reset(None)

    # ------------------------------------------------------------------ state
    def reset(self, t):
        self.started = False
        self.lap_start = None
        self.next_idx = 0
        self.splits = {}
        self.missed = []

    @property
    def next_checkpoint(self):
        if not self.started:
            return self.env.element_ids[0]          # the timing line
        if self.next_idx < len(self.seq):
            return self.seq[self.next_idx]
        return self.env.element_ids[0]

    def valid_laps(self):
        return [l for l in self.laps if l["valid"]]

    def best_lap(self):
        v = self.valid_laps()
        return min(v, key=lambda l: l["time_s"]) if v else None

    # ------------------------------------------------------------------ events
    def on_crossing(self, elem_index: int, t: float, lat: float, z: float, direction: float):
        eid = self.env.element_ids[int(elem_index)]
        fwd = direction > 0
        rec = {"t": t, "element": eid, "lateral_m": lat, "z_m": z, "direction": int(direction),
               "counted": False}
        self.crossings.append(rec)
        events = []
        if self.require_forward and not fwd:
            return events
        if elem_index == 0:
            if self.started and self.next_idx > 0:
                lap = self._finish(t)
                events.append(("lap", lap))
            elif self.started:
                events.append(("lap_restart", {"t": t}))
            else:
                events.append(("lap_start", {"t": t}))
            self.started, self.lap_start, self.next_idx = True, t, 0
            self.splits, self.missed = {}, []
            rec["counted"] = True
            return events
        if not self.started:
            return events
        cp = self.env.checkpoint_of(eid)
        if cp is None:
            return events
        remaining = self.seq[self.next_idx:]
        if cp not in remaining:
            return events
        k = remaining.index(cp)
        for skipped in remaining[:k]:
            self.missed.append(skipped)
            events.append(("gate_missed", {"t": t, "checkpoint": skipped}))
        self.next_idx += k + 1
        self.splits[cp] = t - self.lap_start
        rec["counted"] = True
        events.append(("gate", {"t": t, "checkpoint": cp, "element": eid,
                                "split_s": t - self.lap_start, "lateral_m": lat, "z_m": z}))
        return events

    def _finish(self, t):
        missed = self.missed + self.seq[self.next_idx:]
        lap = {"lap": len(self.laps) + 1, "t_start": self.lap_start, "t_end": t,
               "time_s": t - self.lap_start,
               "splits_s": {cp: self.splits.get(cp) for cp in self.seq},
               "passed": [cp for cp in self.seq if cp in self.splits],
               "missed": missed, "valid": not missed}
        self.laps.append(lap)
        return lap

    def abort(self, t, reason):
        """Reset / crash mid-lap: the partial lap is discarded but recorded."""
        info = None
        if self.started:
            info = {"t": t, "reason": reason, "elapsed_s": t - self.lap_start,
                    "passed": [cp for cp in self.seq if cp in self.splits]}
        self.reset(t)
        return info
