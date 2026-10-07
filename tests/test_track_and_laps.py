"""Track loading (timing line != gate), in-aperture crossings, gate collisions, lap logic."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpvsim import config as C  # noqa: E402
from fpvsim import dynamics as D  # noqa: E402
from fpvsim import track_env  # noqa: E402
from fpvsim.laps import LapTracker  # noqa: E402
from fly import argv_for_track  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return C.load_config()


@pytest.fixture(scope="module")
def env(cfg):
    return track_env.load(cfg)


def test_timing_line_is_not_a_gate(env):
    tl = env.track.timing_line
    assert tl.eid == "start" and tl.is_timing_line and not tl.is_gate
    assert "start" not in env.gate_ids
    assert "start" not in env.sequence
    assert env.element_ids[0] == "start"
    assert len(env.G) == len(env.gate_ids) == 11          # 10 stations, 11 openings
    assert env.E[0, 5] == pytest.approx(1.173)


def test_saved_track_catalog_and_open_world():
    presets = C.load_track_presets()
    assert [preset["id"] for preset in presets] == ["neon-circuit", "open-world"]
    cfg = C.load_config(track="open-world")
    assert cfg["track"]["preset"] == "open-world"
    assert cfg["track"]["name"] == "Open World Test Range"
    assert cfg["track"]["ceiling_height_m"] is None
    assert cfg["track"]["venue_walls"] is False
    open_env = track_env.load(cfg)
    assert len(open_env.G) == 6
    P = D.build_params(cfg)
    D.configure_venue_bounds(P, open_env.bbox, cfg)
    assert P[D.P_WALL_ON] == 0.0
    with pytest.raises(ValueError, match="unknown track preset"):
        C.load_config(track="missing-track")


def test_track_switch_arguments_are_replaced():
    args = ["--keyboard", "--track", "neon-circuit", "--set", "display.vsync=true"]
    assert argv_for_track(args, "open-world") == [
        "--keyboard", "--set", "display.vsync=true", "--track", "open-world"]
    assert argv_for_track(["--track=neon-circuit"], "open-world") == ["--track", "open-world"]


def test_double_gate_modes(cfg):
    e1 = track_env.load(cfg)
    assert e1.sequence.count("8") == 1 and "8b" not in e1.sequence
    assert len(e1.sequence) == 10
    assert e1.checkpoint_of("8b") == "8" and e1.checkpoint_of("8") == "8"
    c2 = C.load_config(sets=["track.double_gate_mode=sequential"])
    e2 = track_env.load(c2)
    i = e2.sequence.index("8b")
    assert e2.sequence[i + 1] == "8" and len(e2.sequence) == 11


def test_spawn_behind_timing_line(env):
    tl = env.track.timing_line
    n = np.array([math.cos(tl.yaw), math.sin(tl.yaw)])
    d = (np.array(env.spawn_pos[:2]) - np.array(tl.pos[:2])) @ n
    assert d == pytest.approx(-2.0)
    first = env.track.elements[[e.eid for e in env.track.elements].index(env.sequence[0])]
    assert (np.array(first.pos[:2]) - np.array(tl.pos[:2])) @ n > 0   # first gate is ahead


def fly_straight(cfg, env, start, vel, seconds=1.0, mode="reset"):
    c2 = C.load_config(sets=["physics.gravity=0", "physics.drag_linear=[0,0,0]",
                             "physics.drag_quadratic=[0,0,0]", f"collision.mode={mode}"])
    P = D.build_params(c2)
    s = D.new_state(start, 0.0)
    s[D.S_V:D.S_V + 3] = vel
    c = D.new_ctrl()
    dprev = np.zeros(len(env.E))
    D.init_crossings(s, env.E, dprev)
    xb = np.zeros((64, 5))
    L, T = np.zeros((1, D.NLOG), np.float32), np.zeros(1)
    done, ev, nx = D.step_n(s, c, P, env.G, env.E, dprev, np.zeros(3), 0.0, False,
                            int(seconds / P[D.P_DT]), 0.0, L, T, -1, xb)
    return ev, [(env.element_ids[int(r[0])], r[1], r[2], r[3], r[4]) for r in xb[:nx]], s


def through(g, lateral, z, speed=5.0, back=2.0):
    n = np.array([math.cos(g.yaw), math.sin(g.yaw), 0])
    a = np.array([-math.sin(g.yaw), math.cos(g.yaw), 0])
    start = np.array(g.pos) - n * back + a * lateral
    start[2] = g.pos[2] + z
    return start, n * speed


def gate(env, eid):
    return next(e for e in env.track.elements if e.eid == eid)


def test_crossing_inside_aperture_counts(cfg, env):
    g = gate(env, "5")
    ev, xs, _ = fly_straight(cfg, env, *through(g, 0.3, 1.35))
    assert ev == 0
    hits = [x for x in xs if x[0] == "5"]
    assert len(hits) == 1
    _, t, lat, z, d = hits[0]
    assert t == pytest.approx(0.4, abs=2e-3) and lat == pytest.approx(0.3, abs=1e-6)
    assert z == pytest.approx(1.35) and d == 1.0
    ev, xs, _ = fly_straight(cfg, env, *through(g, 0.3, 1.35, speed=-5.0, back=-2.0))
    assert [x for x in xs if x[0] == "5"][0][4] == -1.0     # backwards is flagged


def test_passing_outside_frame_is_not_a_pass(cfg, env):
    g = gate(env, "5")
    ev, xs, _ = fly_straight(cfg, env, *through(g, 1.6, 1.35))   # beside the 1.35 m half-width frame
    assert ev == 0 and not [x for x in xs if x[0] == "5"]
    ev, xs, _ = fly_straight(cfg, env, *through(g, 0.0, 3.2))    # over the top
    assert ev == 0 and not [x for x in xs if x[0] == "5"]


@pytest.mark.parametrize("lat,z", [(0.0, 0.3), (0.0, 2.4), (1.05, 1.35), (-1.05, 1.35), (0.0, 2.05)])
def test_hitting_frame_crashes(cfg, env, lat, z):
    g = gate(env, "5")
    ev, xs, _ = fly_straight(cfg, env, *through(g, lat, z))
    assert ev == 2
    assert not [x for x in xs if x[0] == "5"]


def test_bounce_mode_does_not_reset(cfg, env):
    g = gate(env, "5")
    ev, xs, s = fly_straight(cfg, env, *through(g, 0.0, 0.3), mode="bounce")
    assert ev == 0
    n = np.array([math.cos(g.yaw), math.sin(g.yaw), 0])
    assert (s[3:6] @ n) < 0                                       # bounced back


@pytest.mark.parametrize("axis,side", [(0, -1), (0, 1), (1, -1), (1, 1)])
def test_venue_walls_are_solid(cfg, env, axis, side):
    c2 = C.load_config(sets=["physics.gravity=0", "physics.drag_linear=[0,0,0]",
                             "physics.drag_quadratic=[0,0,0]"])
    P = D.build_params(c2)
    D.configure_venue_bounds(P, env.bbox, c2)
    bounds = ((P[D.P_WALL_XMIN], P[D.P_WALL_XMAX]),
              (P[D.P_WALL_YMIN], P[D.P_WALL_YMAX]))
    start = np.array([(env.bbox[0] + env.bbox[1]) / 2,
                      (env.bbox[2] + env.bbox[3]) / 2, 1.0])
    wall_center = bounds[axis][1 if side > 0 else 0]
    limit = wall_center - side * (P[D.P_WALL_HALF] + P[D.P_RAD])
    start[axis] = limit - side * 0.5
    velocity = np.zeros(3)
    velocity[axis] = side * 5.0
    s = D.new_state(start, 0.0)
    s[D.S_V:D.S_V + 3] = velocity
    c = D.new_ctrl()
    dprev = np.zeros(len(env.E))
    D.init_crossings(s, env.E, dprev)
    done, event, _ = D.step_n(s, c, P, env.G, env.E, dprev, np.zeros(3), 0.0, False,
                              1000, 0.0, np.zeros((1, D.NLOG), np.float32), np.zeros(1),
                              -1, np.zeros((64, 5)))
    assert done < 1000
    assert event == 4
    assert s[axis] == pytest.approx(limit)


def test_ceiling_does_not_exist_outside_venue(cfg, env):
    c2 = C.load_config(sets=["physics.gravity=0"])
    P = D.build_params(c2)
    D.configure_venue_bounds(P, env.bbox, c2)
    outside = (P[D.P_WALL_XMAX] + 2.0, (P[D.P_WALL_YMIN] + P[D.P_WALL_YMAX]) / 2.0,
               P[D.P_CEIL_H] + 1.0)
    s = D.new_state(outside, 0.0)
    c = D.new_ctrl()
    dprev = np.zeros(len(env.E))
    D.init_crossings(s, env.E, dprev)
    done, event, _ = D.step_n(s, c, P, env.G, env.E, dprev, np.zeros(3), 0.0, False,
                              10, 0.0, np.zeros((1, D.NLOG), np.float32), np.zeros(1),
                              -1, np.zeros((64, 5)))
    assert done == 10 and event == 0
    assert s[D.S_P + 2] == pytest.approx(outside[2])


def test_open_venue_walls_can_be_flown_over(cfg, env):
    c2 = C.load_config(sets=["track.ceiling_height_m=null", "physics.gravity=0",
                             "physics.drag_linear=[0,0,0]", "physics.drag_quadratic=[0,0,0]"])
    P = D.build_params(c2)
    D.configure_venue_bounds(P, env.bbox, c2)
    s = D.new_state((P[D.P_WALL_XMAX] - 0.5, 0.0, P[D.P_WALL_H] + P[D.P_RAD] + 0.2), 0.0)
    s[D.S_V] = 5.0
    c = D.new_ctrl()
    dprev = np.zeros(len(env.E))
    D.init_crossings(s, env.E, dprev)
    done, event, _ = D.step_n(s, c, P, env.G, env.E, dprev, np.zeros(3), 0.0, False,
                              300, 0.0, np.zeros((1, D.NLOG), np.float32), np.zeros(1),
                              -1, np.zeros((64, 5)))
    assert done == 300 and event == 0
    assert s[D.S_P] > P[D.P_WALL_XMAX] + P[D.P_WALL_HALF]


def test_double_gate_upper_opening_counts_as_station(cfg, env):
    g = gate(env, "8b")
    ev, xs, _ = fly_straight(cfg, env, *through(g, 0.0, 1.35))
    assert ev == 0 and [x[0] for x in xs if x[0].startswith("8")] == ["8b"]


def idx(env, eid):
    return env.element_ids.index(eid)


def test_lap_tracker(env):
    lt = LapTracker(env)
    assert lt.on_crossing(0, 1.0, 0, 0, -1) == []               # backwards line ignored
    ev = lt.on_crossing(0, 2.0, 0, 0, 1)
    assert ev[0][0] == "lap_start"
    ev = lt.on_crossing(0, 2.5, 0, 0, 1)                         # hovering over the line
    assert ev[0][0] == "lap_restart" and lt.lap_start == 2.5
    t = 3.0
    for cp in env.sequence:
        eid = "8b" if cp == "8" else cp
        ev = lt.on_crossing(idx(env, eid), t, 0, 1.3, 1)
        assert ev[-1][0] == "gate"
        t += 1.0
    ev = lt.on_crossing(0, t, 0, 0, 1)
    assert ev[0][0] == "lap"
    lap = ev[0][1]
    assert lap["valid"] and lap["time_s"] == pytest.approx(t - 2.5) and not lap["missed"]
    assert lap["splits_s"]["1"] == pytest.approx(0.5)
    # second lap skips gate 2 and ignores a backward crossing of gate 3
    lt.on_crossing(idx(env, "1"), t + 1, 0, 1.3, 1)
    assert lt.on_crossing(idx(env, "3"), t + 2, 0, 1.3, -1) == []
    ev = lt.on_crossing(idx(env, "3"), t + 3, 0, 1.3, 1)
    assert ("gate_missed", "2") in [(k, d.get("checkpoint")) for k, d in ev]
    ev = lt.on_crossing(0, t + 4, 0, 0, 1)
    lap2 = ev[0][1]
    assert not lap2["valid"] and "2" in lap2["missed"] and "9" in lap2["missed"]
    assert lt.best_lap()["lap"] == 1
    assert lt.abort(t + 5, "crash")["reason"] == "crash" and not lt.started
