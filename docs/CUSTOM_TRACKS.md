# Custom tracks

Tracks are YAML files containing one timing line, an ordered set of gates, and
optional visual obstacles. Coordinates use metres in a right-handed, z-up world.
Gate positions are the bottom-centre of each frame and yaw is in radians.

Start a one-off custom course without modifying the default configuration:

```text
python run.py --set track.yaml=/absolute/path/to/my_track.yaml
```

## Saved track presets

Plain `python run.py` opens the saved-track selector before flight, and `T` opens it again
during a session. List or launch tracks directly from a terminal with:

```text
python run.py --list-tracks
python run.py --track open-world
```

To add one, save the geometry under `assets/`, then add a YAML file under
`config/tracks/`. Presets are discovered automatically—no Python registry needs to
be edited. A preset contains user-facing metadata and track-specific environment
settings:

```yaml
id: my-track
name: My Track
description: Short description shown in the selector

track:
  yaml: assets/my_track.yaml
  ceiling_height_m: null
  venue_walls: false
  venue_margin_m: 20.0
  venue_wall_thickness_m: 0.4
  venue_wall_height_m: 4.0
  show_venue_features: false
  gate_color_rgb: [0.20, 0.80, 1.00]
```

Use a stable, unique `id`. The selector safely closes the active run and starts a
new one with the chosen preset, so logs and effective configuration snapshots never
mix data from different tracks.

## Minimal example

```yaml
name: "Example Circuit"
footprint: [40.0, 24.0]
gate_width: 2.70
gate_height: 2.70
gate_opening_width: 1.50
gate_opening_height: 1.50
gate_depth: 0.26

lap_order: [start, 1, 2, 3]

timing_line:
  start:
    kind: start_line
    lap_index: 0
    pos: [0.0, 0.0, 0.0]
    yaw: 0.0
    length_m: 2.346

gates:
  "1": {lap_index: 1, pos: [7.0, 0.0, 0.0], yaw: 0.0}
  "2": {lap_index: 2, pos: [12.0, 5.0, 0.0], yaw: 1.570796}
  "3": {lap_index: 3, pos: [5.0, 10.0, 0.0], yaw: 3.141593}
```

The first `lap_order` entry is the timing-line id. Every later entry must identify
a gate. The first gate should be in front of the timing line's normal
`(cos(yaw), sin(yaw), 0)` so the generated spawn faces into the course.

The simulator automatically places solid venue walls outside the outermost course
elements. Adjust `track.venue_margin_m` to control the surrounding flight space;
the rendered and physical walls use the same `track.venue_wall_thickness_m` value.
Set `track.ceiling_height_m` to a height for a finite enclosed roof, or to `null`
for an open venue whose walls can be flown over. `track.venue_wall_height_m`
controls wall height in that open configuration. The roof and walls are finite
3D collision boxes—neither creates an invisible world-wide constraint.
Set `track.venue_walls: false` for a fully open world with no generated wall or roof
geometry. In that mode, `ceiling_height_m` should also be `null`.

## Stacked gates

An id ending in `b` is treated as the upper opening of a stacked gate. For example,
gate `8b` shares the x/y/yaw values of gate `8` and uses a higher z coordinate.
Configure scoring with:

```yaml
track:
  double_gate_mode: one_station  # either opening clears gate 8
  double_gate_sequence: ["8b", "8"]
```

Set `double_gate_mode: sequential` to require upper then lower.

## Current geometry constraint

The renderer, passage scoring, and collision model intentionally share one fixed
gate geometry: 2.70 m outer size, 1.50 m square opening, and 0.26 m depth. The track
loader rejects mismatched dimensions so the visual and physical gates cannot silently
disagree. Supporting per-track gate sizes would require changing all three systems
together and adding geometry tests.

Only publish maps and visual assets you created or have permission to redistribute.
