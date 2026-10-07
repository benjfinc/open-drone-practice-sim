"""Load and validate the simulator's YAML track format."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


GATE_GEOMETRY = {
    "outer_width_m": 2.70,
    "outer_height_m": 2.70,
    "depth_m": 0.26,
    "opening_width_m": 1.50,
    "opening_height_m": 1.50,
}


@dataclass(frozen=True)
class Element:
    eid: str
    kind: str
    pos: tuple
    yaw: float
    lap_index: int
    stacked_upper: bool = False

    @property
    def is_gate(self):
        return self.kind == "gate"

    @property
    def is_timing_line(self):
        return self.kind == "start_line"


@dataclass
class Track:
    name: str
    elements: list = field(default_factory=list)
    footprint: tuple = (48.0, 28.0)
    lap_order: list = field(default_factory=list)
    lap_length_m: float = 0.0
    venue_features: list = field(default_factory=list)

    @property
    def gates(self):
        return [element for element in self.elements if element.is_gate]

    @property
    def timing_line(self):
        lines = [element for element in self.elements if element.is_timing_line]
        if len(lines) != 1:
            raise ValueError(f"expected exactly one timing line, got {len(lines)}")
        return lines[0]

    @property
    def double_gate(self):
        return [element for element in self.elements if element.stacked_upper]


def _element(eid, values, default_kind):
    return Element(
        eid=str(eid),
        kind=values.get("kind", default_kind),
        pos=tuple(float(value) for value in values["pos"]),
        yaw=float(values.get("yaw", 0.0)),
        lap_index=int(values.get("lap_index", -1)),
        stacked_upper=str(eid).endswith("b"),
    )


def load(path, strict=True):
    raw = yaml.safe_load(Path(path).read_text())
    elements = [
        _element(eid, values, "start_line")
        for eid, values in (raw.get("timing_line") or {}).items()
    ]
    elements.extend(
        _element(eid, values, "gate")
        for eid, values in (raw.get("gates") or {}).items()
    )
    features = [
        _element(eid, {**values, "kind": "venue_feature", "lap_index": -1}, "venue_feature")
        for eid, values in (raw.get("obstacles") or {}).items()
    ]
    result = Track(
        name=raw.get("name", "Untitled Track"),
        elements=elements,
        footprint=tuple(float(value) for value in raw.get("footprint", (48.0, 28.0))),
        lap_order=[str(value) for value in raw.get("lap_order", [])],
        lap_length_m=float(raw.get("lap_length_m", 0.0)),
        venue_features=features,
    )
    if strict:
        check(result, raw)
    return result


def check(track, raw):
    errors = []
    geometry_keys = {
        "gate_width": "outer_width_m",
        "gate_height": "outer_height_m",
        "gate_depth": "depth_m",
        "gate_opening_width": "opening_width_m",
        "gate_opening_height": "opening_height_m",
    }
    for yaml_key, geometry_key in geometry_keys.items():
        if yaml_key in raw and abs(float(raw[yaml_key]) - GATE_GEOMETRY[geometry_key]) > 1e-6:
            errors.append(f"{yaml_key} must be {GATE_GEOMETRY[geometry_key]}")
    if len(track.footprint) != 2 or min(track.footprint) <= 0:
        errors.append("footprint must contain two positive dimensions")
    if len([element for element in track.elements if element.is_timing_line]) != 1:
        errors.append("track must define exactly one timing line")
    if not track.gates:
        errors.append("track must define at least one gate")
    ids = [element.eid for element in track.elements]
    if len(ids) != len(set(ids)):
        errors.append("element ids must be unique")
    valid_order_ids = set(ids)
    unknown = [eid for eid in track.lap_order if eid not in valid_order_ids]
    if unknown:
        errors.append(f"lap_order contains unknown elements: {unknown}")
    if errors:
        raise ValueError("invalid track:\n  " + "\n  ".join(errors))
    return True
