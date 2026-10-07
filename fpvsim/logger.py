"""Run logger: 1 kHz tick log + per-frame camera log + events + laps + provenance.

Layout of runs/<timestamp>/  (see README "Log format"):
  ticks.npz      t (float64, s), data (float32 [N, NLOG]), columns
  frames.npz     t, data (float64 [F, len(FRAME_COLUMNS)]), columns
  events.jsonl   one JSON object per event, flushed as it happens
  laps.json      completed laps with splits, passed / missed checkpoints
  laps.csv       one row per lap
  config_used.yaml, calibration_used.yaml, meta.json (git hash, versions, ...)
While running, data is written in chunks under chunks/ so a hard kill loses at
most one chunk; `loader.load_run` reads either form.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from .dynamics import LOG_COLUMNS, NLOG

FRAME_COLUMNS = ["t", "frame", "cam_mode",
                 "cam_px", "cam_py", "cam_pz", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
                 "body_px", "body_py", "body_pz", "body_qw", "body_qx", "body_qy", "body_qz",
                 "tilt_deg", "hfov_deg", "vfov_deg", "wall_time"]


def git_info(repo: Path):
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain", "--", ".") or ""
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "fpv_sim_uncommitted_files": len([l for l in status.splitlines() if l.strip()]),
            "note": "fpv_sim is not committed; its source is snapshotted in meta.json source_md5"}


class RunLogger:
    def __init__(self, root: Path, chunk_ticks: int = 60000, frame_chunk: int = 20000):
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.chunks = self.dir / "chunks"
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.cap = int(chunk_ticks)
        self.L = np.zeros((self.cap, NLOG), np.float32)
        self.T = np.zeros(self.cap, np.float64)
        self.rows = 0
        self.nchunk = 0
        self.frames = []
        self.frame_chunk = frame_chunk
        self.nfchunk = 0
        self.ev = open(self.dir / "events.jsonl", "a")
        self.closed = False

    # --------------------------------------------------------------- ticks
    def reserve(self, n: int) -> int:
        """Row index where the next n ticks go (flushes a full chunk first)."""
        if n > self.cap:
            raise ValueError("batch larger than chunk")
        if self.rows + n > self.cap:
            self.flush_ticks()
        return self.rows

    def commit(self, n: int):
        self.rows += n

    def flush_ticks(self):
        if self.rows:
            np.save(self.chunks / f"t_{self.nchunk:04d}.npy", self.T[:self.rows])
            np.save(self.chunks / f"ticks_{self.nchunk:04d}.npy", self.L[:self.rows])
            self.nchunk += 1
            self.rows = 0

    # --------------------------------------------------------------- frames
    def frame(self, row):
        self.frames.append(row)
        if len(self.frames) >= self.frame_chunk:
            self.flush_frames()

    def flush_frames(self):
        if self.frames:
            np.save(self.chunks / f"frames_{self.nfchunk:04d}.npy", np.asarray(self.frames, np.float64))
            self.nfchunk += 1
            self.frames = []

    # --------------------------------------------------------------- events
    def event(self, kind: str, **kw):
        rec = {"kind": kind, **kw}
        self.ev.write(json.dumps(rec, default=float) + "\n")
        self.ev.flush()

    # --------------------------------------------------------------- provenance
    def write_meta(self, cfg, calib, calib_path, repo: Path, extra=None):
        (self.dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (self.dir / "calibration_used.yaml").write_text(
            f"# source: {calib_path}\n" + yaml.safe_dump(calib, sort_keys=False))
        src = {}
        here = Path(__file__).resolve().parents[1]
        for p in sorted(list(here.glob("fpvsim/*.py")) + list(here.glob("*.py"))):
            src[str(p.relative_to(here))] = hashlib.md5(p.read_bytes()).hexdigest()
        track_yaml = Path(cfg["track"]["yaml"])
        track_yaml = track_yaml if track_yaml.is_absolute() else repo / track_yaml
        meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info(repo),
                "python": platform.python_version(), "numpy": np.__version__,
                "track_yaml": str(track_yaml),
                "track_yaml_md5": hashlib.md5(track_yaml.read_bytes()).hexdigest(),
                "source_md5": src, "tick_columns": LOG_COLUMNS, "frame_columns": FRAME_COLUMNS,
                "conventions": {
                    "world": "track map frame, metres, z up",
                    "body": "FLU: x forward, y left, z up",
                    "quaternion": "[w,x,y,z] Hamilton, body->world",
                    "camera_frame": "OpenGL/Isaac: +X right, +Y up, -Z forward; cam quat [w,x,y,z] camera->world",
                    "omega": "body FLU rad/s", "velocity": "world m/s",
                    "setpoints": "body FLU rad/s (sp_raw = from rates curve, sp = after RC smoothing)",
                    "sticks": "roll +right, pitch +forward, yaw +right, thr 0..1",
                    "motors": "Betaflight order 1 RR, 2 FR, 3 RL, 4 FL; mcmd 0..1, thrust N"}}
        if extra:
            meta.update(extra)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # --------------------------------------------------------------- close
    def close(self, laps, aborted_laps=(), crossings=()):
        if self.closed:
            return
        self.closed = True
        self.flush_ticks()
        self.flush_frames()
        self.ev.close()
        (self.dir / "laps.json").write_text(json.dumps(
            {"laps": laps, "aborted": list(aborted_laps), "crossings": list(crossings)},
            indent=1, default=float))
        with open(self.dir / "laps.csv", "w", newline="") as f:
            w = csv.writer(f)
            seq = list(laps[0]["splits_s"].keys()) if laps else []
            w.writerow(["lap", "time_s", "valid", "n_passed", "missed"] + [f"split_{s}" for s in seq])
            for l in laps:
                w.writerow([l["lap"], f"{l['time_s']:.3f}", l["valid"], len(l["passed"]),
                            " ".join(l["missed"])]
                           + ["" if l["splits_s"][s] is None else f"{l['splits_s'][s]:.3f}" for s in seq])
        # consolidate chunks
        tk = sorted(self.chunks.glob("ticks_*.npy"))
        if tk:
            data = np.concatenate([np.load(p) for p in tk])
            t = np.concatenate([np.load(p) for p in sorted(self.chunks.glob("t_*.npy"))])
            np.savez(self.dir / "ticks.npz", t=t, data=data, columns=np.array(LOG_COLUMNS))
        fr = sorted(self.chunks.glob("frames_*.npy"))
        if fr:
            data = np.concatenate([np.load(p) for p in fr])
            np.savez(self.dir / "frames.npz", data=data, columns=np.array(FRAME_COLUMNS))
        shutil.rmtree(self.chunks, ignore_errors=True)
