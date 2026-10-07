"""YAML config loading with dotted command-line overrides."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]          # repo root
REPO = HERE
DEFAULT_CONFIG = HERE / "config" / "default.yaml"
DEFAULT_CALIB = HERE / "config" / "calibration.yaml"
FALLBACK_CALIB = HERE / "config" / "calibration_default.yaml"
TRACK_PRESET_DIR = HERE / "config" / "tracks"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_set(cfg: dict, expr: str):
    key, _, val = expr.partition("=")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(val)


def load_track_presets(directory=None):
    """Discover and validate saved track preset overlays."""
    preset_dir = Path(directory or TRACK_PRESET_DIR)
    presets = []
    seen = set()
    for preset_path in sorted(preset_dir.glob("*.yaml")):
        raw = yaml.safe_load(preset_path.read_text()) or {}
        preset_id = str(raw.get("id", "")).strip()
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", preset_id) or preset_id in seen:
            raise ValueError(f"invalid or duplicate track preset id in {preset_path}")
        if not raw.get("name") or not raw.get("description"):
            raise ValueError(f"track preset needs name and description: {preset_path}")
        track_cfg = raw.get("track")
        if not isinstance(track_cfg, dict) or not track_cfg.get("yaml"):
            raise ValueError(f"track preset needs track.yaml: {preset_path}")
        track_path = Path(track_cfg["yaml"])
        track_path = track_path if track_path.is_absolute() else REPO / track_path
        if not track_path.is_file():
            raise ValueError(f"track preset geometry does not exist: {track_path}")
        seen.add(preset_id)
        presets.append({
            "id": preset_id,
            "name": str(raw["name"]),
            "description": str(raw["description"]),
            "path": preset_path,
            "track": copy.deepcopy(track_cfg),
        })
    return presets


def track_preset(preset_id, directory=None):
    presets = load_track_presets(directory)
    for preset in presets:
        if preset["id"] == preset_id:
            return preset
    available = ", ".join(preset["id"] for preset in presets) or "none"
    raise ValueError(f"unknown track preset {preset_id!r}; available: {available}")


def load_config(path=None, overlays=(), sets=(), track=None):
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    selected_track = track if track is not None else cfg.get("track", {}).get("preset")
    if selected_track:
        preset = track_preset(str(selected_track))
        cfg = deep_merge(cfg, {"track": preset["track"]})
        cfg["track"]["preset"] = preset["id"]
        cfg["track"]["name"] = preset["name"]
        cfg["track"]["description"] = preset["description"]
    for ov in overlays:
        cfg = deep_merge(cfg, yaml.safe_load(Path(ov).read_text()))
    for s in sets:
        apply_set(cfg, s)
    return cfg


def load_calibration(path=None):
    p = Path(path) if path else (DEFAULT_CALIB if DEFAULT_CALIB.exists() else FALLBACK_CALIB)
    return yaml.safe_load(p.read_text()), p


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p
