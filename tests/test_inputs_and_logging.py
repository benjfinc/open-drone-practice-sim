import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpvsim import config as C  # noqa: E402
from fpvsim import inputs  # noqa: E402
from fpvsim.dynamics import COL, EV_RESET  # noqa: E402
from fpvsim.loader import camera_poses, load_run  # noqa: E402
from fpvsim.logger import RunLogger  # noqa: E402
from fpvsim.render import Renderer, fpv_camera_R_body, mat_to_quat  # noqa: E402
from fpvsim.sim import Sim  # noqa: E402


def test_normalisation():
    a = {"center": 0.1, "min": -0.9, "max": 0.9}
    assert inputs.norm_bipolar(0.1, a) == pytest.approx(0)
    assert inputs.norm_bipolar(0.9, a) == pytest.approx(1)
    assert inputs.norm_bipolar(-0.9, a) == pytest.approx(-1)
    assert inputs.norm_bipolar(0.5, a) == pytest.approx(0.5)
    assert inputs.norm_bipolar(0.9, {**a, "invert": True}) == pytest.approx(-1)
    t = {"min": -1, "max": 1}
    assert inputs.norm_unipolar(-1, t) == 0 and inputs.norm_unipolar(1, t) == 1
    assert inputs.norm_unipolar(0, {**t, "invert": True}) == pytest.approx(0.5)


def test_switches():
    raw = [0, 0, 0, 0, -1.0, 0.02, 1.0]
    assert inputs.switch_on(raw, [], {"kind": "axis", "index": 4, "threshold": 0}) is False
    assert inputs.switch_on(raw, [], {"kind": "axis", "index": 4, "threshold": 0, "invert": True}) is True
    sw3 = {"kind": "axis", "index": 5, "positions": 3}
    assert inputs.switch_position(raw, [], sw3) == 1
    assert inputs.switch_position([0, 0, 0, 0, 0, -1.0], [], sw3) == 0
    assert inputs.switch_position([0, 0, 0, 0, 0, 1.0], [], sw3) == 2
    assert inputs.switch_on(raw, [0, 1], {"kind": "button", "index": 1}) is True


def test_camera_tilt_geometry():
    R = fpv_camera_R_body(20.0)
    fwd = -R[:, 2]
    assert np.degrees(np.arctan2(fwd[2], fwd[0])) == pytest.approx(20.0)   # looks UP by the tilt
    assert R[:, 0] @ np.array([0, -1, 0]) == pytest.approx(1)              # +X right = body -y
    assert np.linalg.det(R) == pytest.approx(1)
    q = mat_to_quat(R)
    from fpvsim.dynamics import quat_to_R_np
    assert np.allclose(quat_to_R_np(q), R, atol=1e-9)


def test_renderer_resize_rebuilds_osd_texture():
    class Texture:
        def __init__(self, size):
            self.size = size
            self.released = False

        def release(self):
            self.released = True

    class Context:
        def __init__(self):
            self.created = []

        def texture(self, size, components):
            assert components == 4
            texture = Texture(size)
            self.created.append(texture)
            return texture

    renderer = Renderer.__new__(Renderer)
    renderer.w, renderer.h = 1280, 720
    renderer.ctx = Context()
    renderer.pg = None
    old_texture = Texture((1280, 720))
    renderer.osd_tex = old_texture

    assert renderer.resize((2560, 1600))
    assert old_texture.released
    assert (renderer.w, renderer.h) == (2560, 1600)
    assert renderer.osd_tex.size == (2560, 1600)
    assert not renderer.resize((2560, 1600))


def test_disarmed_throttle_prompt_is_latched():
    cfg = C.load_config()
    sim = Sim(cfg)

    sim.set_armed(False, 0.2)
    assert sim.active_messages() == ["ARM FIRST — PRESS SPACE"]

    sim.set_armed(False, 0.4)
    assert sim.active_messages() == ["ARM FIRST — PRESS SPACE"]

    sim.set_armed(False, 0.0)
    sim.set_armed(False, 0.2, arm_hint="USE ARM SWITCH")
    assert sim.active_messages()[-1] == "ARM FIRST — USE ARM SWITCH"
    sim.close()


def test_sim_logging_roundtrip(tmp_path):
    cfg = C.load_config()
    lg = RunLogger(tmp_path, chunk_ticks=1500)                    # force several chunks
    sim = Sim(cfg, lg)
    lg.write_meta(cfg, {"verified": False}, "test", C.REPO)
    st = inputs.Sticks()
    sim.set_armed(True, 0.0)
    assert sim.armed
    st.throttle = 0.6
    for _ in range(500):
        sim.advance(8, st)                                        # 4000 ticks
    sim.reset("key")
    sim.advance(1000, st)
    for i in range(10):
        lg.frame([sim.t, i, 0] + [0.0] * 18)
    run_dir = lg.dir
    sim.close()
    r = load_run(run_dir)
    assert len(r.t) == 5000 and r.data.shape[1] == len(r.columns)
    assert np.allclose(np.diff(r.t), 1e-3)
    assert r["armed"].min() == 1.0 and r["stick_thr"][0] == pytest.approx(0.6)
    assert r["pz"][3999] > 1.0                                    # it climbed
    assert (r["event_flags"].astype(int) & EV_RESET).any()
    assert r.segments()[-1] >= 1
    assert r.meta["git"]["commit"] and r.meta["tick_columns"][COL["px"]] == "px"
    assert json.loads((run_dir / "laps.json").read_text())["aborted"] == []
    assert not (run_dir / "chunks").exists() and (run_dir / "ticks.npz").exists()
    cams = camera_poses(r, 30.0)
    assert len(cams["t"]) == pytest.approx(150, abs=2)
    assert np.allclose(np.linalg.norm(cams["cam_quat_wxyz"], axis=1), 1)
