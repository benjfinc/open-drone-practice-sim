"""OpenGL renderer for the course, vehicle, cameras, and on-screen display."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .dynamics import quat_to_R_np

CAM_MODES = ["fpv", "chase", "static"]

VERT = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_model;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec3 in_col;
out vec3 v_nrm; out vec2 v_uv; out vec3 v_col; out vec3 v_wpos;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_wpos = w.xyz; v_nrm = mat3(u_model) * in_nrm; v_uv = in_uv; v_col = in_col;
    gl_Position = u_vp * w;
}
"""
FRAG = """
#version 330
uniform sampler2D u_tex; uniform vec3 u_light; uniform vec3 u_cam;
uniform float u_fog; uniform vec3 u_fogcol; uniform float u_emissive;
in vec3 v_nrm; in vec2 v_uv; in vec3 v_col; in vec3 v_wpos;
out vec4 f_color;
void main() {
    vec3 t = texture(u_tex, v_uv).rgb * v_col;
    float d = abs(dot(normalize(v_nrm), u_light));
    vec3 c = t * (0.50 + 0.50 * d + u_emissive);
    float f = exp(-u_fog * length(v_wpos - u_cam));
    f_color = vec4(mix(u_fogcol, c, f), 1.0);
}
"""
VERT2 = "#version 330\nin vec2 in_pos;\nvoid main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }\n"
FRAG2 = "#version 330\nuniform vec4 u_color; out vec4 f_color;\nvoid main(){ f_color = u_color; }\n"
VERT3 = ("#version 330\nin vec2 in_pos; out vec2 v_uv;\n"
         "void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }\n")
FRAG3 = ("#version 330\nuniform sampler2D u_tex; in vec2 v_uv; out vec4 f_color;\n"
         "void main(){ f_color = texture(u_tex, v_uv); }\n")

FACES = [
    [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
    [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
    [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)],
    [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)],
]


# ---------------------------------------------------------------- math helpers
def perspective(vfov_rad, aspect, near, far):
    f = 1.0 / math.tan(vfov_rad / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]])


def view_matrix(pos, R_wc):
    V = np.eye(4)
    V[:3, :3] = R_wc.T
    V[:3, 3] = -R_wc.T @ np.asarray(pos)
    return V


def mat_to_quat(R):
    """Rotation matrix -> [w,x,y,z]."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(m[i, i] - m[j, j] - m[k, k] + 1.0, 1e-12)) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[1 + i] = s / 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def fpv_camera_R_body(tilt_deg):
    """Columns = camera +X(right), +Y(up), +Z(back) expressed in body FLU."""
    t = math.radians(tilt_deg)
    right = np.array([0.0, -1.0, 0.0])
    up = np.array([-math.sin(t), 0.0, math.cos(t)])
    back = np.array([-math.cos(t), 0.0, -math.sin(t)])
    return np.column_stack([right, up, back])


def fpv_camera_pose(body_pos, body_quat, tilt_deg, offset_body):
    R_wb = quat_to_R_np(body_quat)
    R_wc = R_wb @ fpv_camera_R_body(tilt_deg)
    return np.asarray(body_pos) + R_wb @ np.asarray(offset_body, float), R_wc


def look_at(eye, target, up=(0, 0, 1)):
    back = np.asarray(eye, float) - np.asarray(target, float)
    back /= max(np.linalg.norm(back), 1e-9)
    right = np.cross(np.asarray(up, float), back)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(back, right), back])


def hfov_to_vfov(hfov_deg, aspect):
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))


def spec_hfov_deg(cam_cfg):
    return math.degrees(2 * math.atan(cam_cfg["spec_width"] / 2 / cam_cfg["spec_fx"]))


# ---------------------------------------------------------------- geometry
def _v(p, n, uv, col):
    return [p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1], col[0], col[1], col[2]]


def box_verts(center, half, col, R=None, uv_per_m=None):
    R = np.eye(3) if R is None else R
    c, h = np.asarray(center, float), np.asarray(half, float)
    out = []
    for face in FACES:
        corners = [c + R @ (np.array(sgn) * h) for sgn in face]
        n = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        n = n / max(np.linalg.norm(n), 1e-12)
        if uv_per_m:
            wu = np.linalg.norm(corners[1] - corners[0]) * uv_per_m
            wv = np.linalg.norm(corners[3] - corners[0]) * uv_per_m
        else:
            wu = wv = 1.0
        uvs = [(0, 0), (wu, 0), (wu, wv), (0, wv)]
        for k in (0, 1, 2, 0, 2, 3):
            out.append(_v(corners[k], n, uvs[k], col))
    return out


def floor_texture(n=1024, seed=0):
    rng = np.random.default_rng(seed)
    def smooth(res, amp):
        small = Image.fromarray((rng.random((res, res)) * 255).astype(np.uint8))
        return (np.asarray(small.resize((n, n), Image.BICUBIC), float) / 255 - 0.5) * amp
    base = 0.40 + smooth(8, 0.10) + smooth(64, 0.06) + smooth(256, 0.05)
    base += (rng.random((n, n)) - 0.5) * 0.05
    yy, xx = np.mgrid[0:n, 0:n]
    base += np.where(((xx * 2 // n) + (yy * 2 // n)) % 2 == 0, 0.035, -0.035)
    cell = n // 4                                          # 1 m lines when a tile is 4 m
    line = ((xx % cell) < 3) | ((yy % cell) < 3)
    base = np.where(line, base * 0.55, base)
    edge = (xx < 6) | (yy < 6)
    base = np.where(edge, 0.85, base)
    rgb = np.stack([base * 0.92, base * 0.97, base * 1.05], -1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def checker_texture(nx=16, ny=2, px=16):
    a = np.indices((ny * px, nx * px))
    c = ((a[0] // px + a[1] // px) % 2) * 235 + 10
    return np.repeat(c[..., None], 3, -1).astype(np.uint8)


class Renderer:
    def __init__(self, ctx, cfg, env, size, pygame=None):
        import moderngl
        self.mgl = moderngl
        self.ctx, self.cfg, self.env = ctx, cfg, env
        self.w, self.h = size
        self.cam_cfg = cfg["camera"]
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.prog2 = ctx.program(vertex_shader=VERT2, fragment_shader=FRAG2)
        self.prog3 = ctx.program(vertex_shader=VERT3, fragment_shader=FRAG3)
        self.fogcol = (0.55, 0.62, 0.70)
        self.batches = []
        self._build_scene()
        self._build_drone()
        self.line_vbo = ctx.buffer(reserve=4 * 2 * 512)
        self.line_vao = ctx.vertex_array(self.prog2, [(self.line_vbo, "2f", "in_pos")])
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], "f4")
        self.osd_vao = ctx.vertex_array(self.prog3, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.osd_tex = ctx.texture((self.w, self.h), 4)
        self.pg = pygame
        self.font = self.font_big = None
        if pygame is not None:
            self._build_fonts()
        self.chase_pos = None
        st = self.cam_cfg.get("static_pos")
        xmin, xmax, ymin, ymax = env.bbox
        ceil_h = cfg["track"].get("ceiling_height_m")
        z_static = 9.0 if ceil_h is None else min(9.0, float(ceil_h) - 0.4)   # stay under the ceiling
        self.static_pos = np.array(st if st else [xmin - 8.0, ymin - 8.0, z_static], float)

    def _build_fonts(self):
        self.pg.font.init()
        # Scale against both dimensions so text cannot overwhelm a narrow window.
        fs = max(10, min(30, self.h // 34, self.w // 64))
        family = "dejavusansmono,liberationmono,monospace"
        self.font = self.pg.font.SysFont(family, fs, bold=True)
        self.font_big = self.pg.font.SysFont(family, fs * 2, bold=True)

    def resize(self, size):
        """Resize the viewport-dependent resources after a display-mode change."""
        w, h = (max(1, int(value)) for value in size)
        if (w, h) == (self.w, self.h):
            return False
        self.w, self.h = w, h
        self.osd_tex.release()
        self.osd_tex = self.ctx.texture((self.w, self.h), 4)
        if self.pg is not None:
            self._build_fonts()
        return True

    # ---------------------------------------------------------------- scene
    def _tex(self, arr, repeat=True, mip=True):
        arr = np.ascontiguousarray(np.flipud(arr))
        t = self.ctx.texture((arr.shape[1], arr.shape[0]), arr.shape[2], arr.tobytes())
        t.repeat_x = t.repeat_y = repeat
        if mip:
            t.build_mipmaps()
            t.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
            t.anisotropy = 16.0
        return t

    def _add(self, verts, tex, emissive=0.0):
        a = np.asarray(verts, "f4")
        vbo = self.ctx.buffer(a.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 2f 3f", "in_pos", "in_nrm", "in_uv", "in_col")])
        self.batches.append((vao, tex, emissive, len(a)))
        return vao

    def _build_scene(self):
        env, tc = self.env, self.cfg["track"]
        white = self.ctx.texture((1, 1), 3, b"\xff\xff\xff")
        self.white = white
        # floor
        ftex = self._tex(floor_texture())
        xmin, xmax, ymin, ymax = env.bbox
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        S, tile = 150.0, 4.0
        corners = [(cx - S, cy - S), (cx + S, cy - S), (cx + S, cy + S), (cx - S, cy + S)]
        fl = []
        for k in (0, 1, 2, 0, 2, 3):
            x, y = corners[k]
            fl.append(_v((x, y, 0.0), (0, 0, 1), (x / tile, y / tile), (1, 1, 1)))
        self._add(fl, ftex)
        # Venue walls share their dimensions with the physical boundary planes.
        # With a ceiling, the walls reach exactly up to it.
        ceil_h = tc.get("ceiling_height_m")
        m = float(tc.get("venue_margin_m", 12.0))
        wall_half = float(tc.get("venue_wall_thickness_m", 0.4)) / 2.0
        H = float(tc.get("venue_wall_height_m", 8.0) if ceil_h is None else ceil_h)
        wcol = (0.75, 0.72, 0.68)
        if tc.get("venue_walls", True):
            wall = []
            for c_, h_ in [(((xmin + xmax) / 2, ymin - m, H / 2), ((xmax - xmin) / 2 + m, wall_half, H / 2)),
                           (((xmin + xmax) / 2, ymax + m, H / 2), ((xmax - xmin) / 2 + m, wall_half, H / 2)),
                           ((xmin - m, (ymin + ymax) / 2, H / 2), (wall_half, (ymax - ymin) / 2 + m, H / 2)),
                           ((xmax + m, (ymin + ymax) / 2, H / 2), (wall_half, (ymax - ymin) / 2 + m, H / 2))]:
                wall += box_verts(c_, h_, wcol, uv_per_m=0.25)
            # a few pillars inside the hall for parallax (well outside the course)
            for px in (xmin - m + 3, xmax + m - 3):
                for py in np.linspace(ymin - m + 3, ymax + m - 3, 5):
                    wall += box_verts((px, py, H / 2), (0.3, 0.3, H / 2), (0.55, 0.55, 0.58), uv_per_m=0.5)
            self._add(wall, ftex)
        # Ceiling is a finite box, matching the physics instead of an infinite plane.
        if tc.get("venue_walls", True) and ceil_h is not None:
            cz = float(ceil_h)
            ceiling = box_verts((cx, cy, cz + wall_half),
                                ((xmax - xmin) / 2 + m, (ymax - ymin) / 2 + m, wall_half),
                                (0.62, 0.64, 0.68), uv_per_m=0.5)
            self._add(ceiling, ftex)
        if tc.get("show_venue_features"):
            vf = []
            for v in env.track.venue_features:
                vf += box_verts((v.pos[0], v.pos[1], 0.5), (1.05, 1.05, 0.5), (0.3, 0.5, 0.8))
            self._add(vf, white)
        # timing line: floor stripe along the line, checker pattern
        tl = env.track.timing_line
        n = np.array([math.cos(tl.yaw), math.sin(tl.yaw), 0.0])
        a = np.array([-math.sin(tl.yaw), math.cos(tl.yaw), 0.0])
        c = np.array([tl.pos[0], tl.pos[1], 0.02])
        hl, hw = 1.173, 0.15
        q = [c - a * hl - n * hw, c + a * hl - n * hw, c + a * hl + n * hw, c - a * hl + n * hw]
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self._add([_v(q[k], (0, 0, 1), uvs[k], (1, 1, 1)) for k in (0, 1, 2, 0, 2, 3)],
                  self._tex(checker_texture(), repeat=False, mip=True))
        # Gates are generated from the same collision boxes used by the physics.
        from .dynamics import GATE_BOXES
        local = []
        for box in GATE_BOXES:
            for vertex in box_verts(box[:3], box[3:], (1, 1, 1)):
                local.append((np.array(vertex[0:3]), np.array(vertex[3:6]), np.array(vertex[6:8])))
        color = np.asarray(tc.get("gate_color_rgb", (0.96, 0.30, 0.08)), float)
        color = tuple(int(value) for value in (np.clip(color, 0.0, 1.0) * 255))
        gtex = self._tex(np.full((4, 4, 3), color, np.uint8), mip=False)
        self.gate_source = f"procedural solid RGB {color}"
        gv = []
        for g in env.track.gates:
            co, si = math.cos(g.yaw), math.sin(g.yaw)
            R = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
            p0 = np.asarray(g.pos, float)
            for p, nrm, uv in local:
                gv.append(_v(p0 + R @ p, R @ nrm, uv, (1, 1, 1)))
        self._add(gv, gtex, emissive=0.35)

    def _build_drone(self):
        v = box_verts((0, 0, 0), (0.07, 0.035, 0.018), (0.12, 0.12, 0.14))
        a = 0.078
        for ang in (45, -45):
            t = math.radians(ang)
            R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            v += box_verts((0, 0, 0), (a * 1.45, 0.009, 0.005), (0.2, 0.2, 0.2), R=R)
        for (x, y) in [(a, a), (a, -a), (-a, a), (-a, -a)]:
            col = (0.95, 0.15, 0.1) if x > 0 else (0.1, 0.8, 0.3)
            v += box_verts((x, y, 0.02), (0.06, 0.06, 0.003), col)
        a_ = np.asarray(v, "f4")
        self.drone_vao = self.ctx.vertex_array(
            self.prog, [(self.ctx.buffer(a_.tobytes()), "3f 3f 2f 3f", "in_pos", "in_nrm", "in_uv", "in_col")])
        self.drone_n = len(a_)

    # ---------------------------------------------------------------- camera
    def camera(self, s, mode, tilt_deg):
        p, q = s[0:3], s[6:10]
        if mode == "fpv":
            return fpv_camera_pose(p, q, tilt_deg, self.cam_cfg["offset_body_m"])
        R_wb = quat_to_R_np(q)
        if mode == "chase":
            f = R_wb[:, 0].copy()
            f[2] = 0
            f = f / np.linalg.norm(f) if np.linalg.norm(f) > 1e-3 else np.array([1.0, 0, 0])
            want = p - f * self.cam_cfg["chase_distance_m"] + np.array([0, 0, self.cam_cfg["chase_height_m"]])
            self.chase_pos = want if self.chase_pos is None else self.chase_pos + 0.15 * (want - self.chase_pos)
            return self.chase_pos.copy(), look_at(self.chase_pos, p + f * 1.0)
        return self.static_pos.copy(), look_at(self.static_pos, p)

    # ---------------------------------------------------------------- draw
    def render(self, fbo, s, mode, tilt_deg, hfov_deg, target_world=None, osd=None):
        ctx = self.ctx
        fbo.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(*self.fogcol, 1.0)
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.disable(self.mgl.CULL_FACE)
        ctx.disable(self.mgl.BLEND)
        aspect = self.w / self.h
        vfov = hfov_to_vfov(hfov_deg, aspect)
        cam_p, R_wc = self.camera(s, mode, tilt_deg)
        Pm = perspective(math.radians(vfov), aspect, self.cam_cfg["near_m"], self.cam_cfg["far_m"])
        VP = Pm @ view_matrix(cam_p, R_wc)
        pr = self.prog
        pr["u_vp"].write(VP.T.astype("f4").tobytes())
        pr["u_model"].write(np.eye(4, dtype="f4").tobytes())
        L = np.array([0.35, 0.25, 0.9])
        pr["u_light"].value = tuple(L / np.linalg.norm(L))
        pr["u_cam"].value = tuple(cam_p)
        pr["u_fog"].value = float(self.cfg["display"]["fog_density"])
        pr["u_fogcol"].value = self.fogcol
        for vao, tex, em, n in self.batches:
            tex.use(0)
            pr["u_tex"].value = 0
            pr["u_emissive"].value = em
            vao.render(self.mgl.TRIANGLES)
        if mode != "fpv":
            M = np.eye(4)
            M[:3, :3] = quat_to_R_np(s[6:10])
            M[:3, 3] = s[0:3]
            pr["u_model"].write(M.T.astype("f4").tobytes())
            self.white.use(0)
            pr["u_emissive"].value = 0.2
            self.drone_vao.render(self.mgl.TRIANGLES)
        # 2-D overlay lines: crosshair + next-gate indicator
        ctx.disable(self.mgl.DEPTH_TEST)
        ctx.enable(self.mgl.BLEND)
        ctx.blend_func = self.mgl.SRC_ALPHA, self.mgl.ONE_MINUS_SRC_ALPHA
        lines = []
        if mode == "fpv":
            k = 0.02
            lines += [(-k, 0), (k, 0), (0, -k * aspect), (0, k * aspect)]
        if target_world is not None:
            lines += self._indicator(VP, np.asarray(target_world, float), aspect)
        if lines:
            arr = np.asarray(lines, "f4")[:512]
            self.line_vbo.write(arr.tobytes())
            self.prog2["u_color"].value = (0.2, 1.0, 0.35, 0.9)
            self.line_vao.render(self.mgl.LINES, vertices=len(arr))
        if osd is not None:
            self.osd_tex.use(0)
            self.prog3["u_tex"].value = 0
            self.osd_vao.render(self.mgl.TRIANGLES)
        cam_q = mat_to_quat(R_wc)
        return cam_p, cam_q, vfov

    def _indicator(self, VP, pw, aspect):
        c = VP @ np.append(pw, 1.0)
        out = []
        if c[3] > 1e-3:
            x, y = c[0] / c[3], c[1] / c[3]
            if abs(x) < 0.97 and abs(y) < 0.97:
                r = 0.035
                pts = [(x, y + r * aspect), (x + r, y), (x, y - r * aspect), (x - r, y)]
                for i in range(4):
                    out += [pts[i], pts[(i + 1) % 4]]
                return out
        else:
            x, y = -c[0], -c[1]                     # behind: mirror the direction
        d = np.array([x, y / aspect])
        if np.linalg.norm(d) < 1e-6:
            d = np.array([0.0, -1.0])
        d /= np.linalg.norm(d)
        tip = d * 0.85
        base = d * 0.72
        perp = np.array([-d[1], d[0]]) * 0.05
        to = lambda v: (float(v[0]), float(v[1] * aspect))
        out += [to(tip), to(base + perp), to(tip), to(base - perp), to(base + perp), to(base - perp)]
        return out

    # ---------------------------------------------------------------- OSD
    def update_osd(self, info):
        """info: dict with tl/tr/bl lists of strings, 'center' string, 'top' string, 'thr' 0..1."""
        pg = self.pg
        if pg is None:
            return
        surf = pg.Surface((self.w, self.h), pg.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        f = self.font
        lh = f.get_linesize()
        white, green, red, yellow = (255, 255, 255), (80, 255, 120), (255, 70, 60), (255, 220, 60)

        def text(s, x, y, col=white, font=f, anchor="tl"):
            img = font.render(s, True, col)
            sh = font.render(s, True, (0, 0, 0))
            w_, h_ = img.get_size()
            if anchor == "tr":
                x -= w_
            elif anchor == "tc":
                x -= w_ // 2
            elif anchor == "c":
                x -= w_ // 2
                y -= h_ // 2
            surf.blit(sh, (x + 2, y + 2))
            surf.blit(img, (x, y))

        m = 12
        for i, s in enumerate(info.get("tl", [])):
            text(s, m, m + i * lh)
        for i, s in enumerate(info.get("tr", [])):
            text(s, self.w - m, m + i * lh, anchor="tr")
        bl = info.get("bl", [])
        for i, s in enumerate(bl):
            col = green if s == "ARMED" else (red if s == "DISARMED" else white)
            text(s, m, self.h - m - (len(bl) - i) * lh, col)
        if info.get("top"):
            text(info["top"], self.w // 2, m, yellow, anchor="tc")
        if info.get("center"):
            text(info["center"], self.w // 2, self.h // 2 - 3 * lh, info.get("center_color", red),
                 font=self.font_big, anchor="c")
        for i, s in enumerate(info.get("msgs", [])):
            text(s, self.w // 2, self.h - m - (i + 2) * lh, yellow, anchor="tc")
        thr = info.get("thr")
        if thr is not None:
            bw, bh = 16, self.h // 4
            x0, y0 = self.w - m - bw, self.h - m - bh
            pg.draw.rect(surf, (0, 0, 0, 140), (x0, y0, bw, bh))
            fh = int(bh * max(0.0, min(1.0, thr)))
            pg.draw.rect(surf, (80, 255, 120, 220), (x0, y0 + bh - fh, bw, fh))
        help_columns = info.get("help_columns")
        if help_columns:
            panel_w = min(self.w - 24, max(320, int(self.w * 0.88)))
            candidate_col_w = panel_w // len(help_columns)

            def required_width(column, col_w):
                margin = max(16, col_w // 28)
                gap = max(18, col_w // 32)
                keys = [key for _, rows in column for key, _ in rows]
                descriptions = [description for _, rows in column for _, description in rows]
                return (2 * margin + max(self.font.size(key)[0] for key in keys) + gap
                        + max(self.font.size(description)[0] for description in descriptions))

            if any(required_width(column, candidate_col_w) > candidate_col_w
                   for column in help_columns):
                sections = [section for column in help_columns for section in column]
                layout_columns = [sections]
            else:
                layout_columns = help_columns

            def column_height(column):
                height = self.font_big.get_linesize() + 20
                for _, rows in column:
                    height += lh + 4 + len(rows) * lh + lh // 2
                return height

            content_h = max(column_height(column) for column in layout_columns)
            panel_h = min(self.h - 32, content_h + lh + 10)
            x0 = (self.w - panel_w) // 2
            y0 = (self.h - panel_h) // 2
            panel = pg.Surface((panel_w, panel_h), pg.SRCALPHA)
            panel.fill((8, 12, 18, 225))
            pg.draw.rect(panel, (255, 120, 35, 255), panel.get_rect(), width=3)
            surf.blit(panel, (x0, y0))
            text("CONTROLS", self.w // 2, y0 + 12, yellow, font=self.font_big, anchor="tc")
            col_w = panel_w // len(layout_columns)
            key_color = (255, 150, 55)
            for col_index, sections in enumerate(layout_columns):
                col_x = x0 + max(16, col_w // 28) + col_index * col_w
                key_x = col_x
                widest_key = max(self.font.size(key)[0] for _, rows in sections for key, _ in rows)
                desc_x = col_x + widest_key + max(18, col_w // 32)
                y = y0 + self.font_big.get_linesize() + 20
                for heading, rows in sections:
                    text(heading, col_x, y, yellow)
                    y += lh + 4
                    for key, description in rows:
                        text(key, key_x, y, key_color)
                        text(description, desc_x, y, white)
                        y += lh
                    y += lh // 2
            text("TAB / H TO CLOSE", self.w // 2, y0 + panel_h - lh - 10,
                 (185, 195, 205), anchor="tc")
        track_menu = info.get("track_menu")
        if track_menu:
            panel_w = min(self.w - 24, max(360, int(self.w * 0.72)))
            panel_h = min(self.h - 32, max(230, int(self.h * 0.72)))
            x0 = (self.w - panel_w) // 2
            y0 = (self.h - panel_h) // 2
            panel = pg.Surface((panel_w, panel_h), pg.SRCALPHA)
            panel.fill((8, 12, 18, 238))
            pg.draw.rect(panel, (45, 205, 230, 255), panel.get_rect(), width=3)
            surf.blit(panel, (x0, y0))
            text("SELECT TRACK", self.w // 2, y0 + 12, yellow, font=self.font_big, anchor="tc")
            text("UP / DOWN TO CHOOSE", self.w // 2,
                 y0 + self.font_big.get_linesize() + 14, (185, 195, 205), anchor="tc")

            items = track_menu["items"]
            selected = int(track_menu["selected"])
            row_h = 2 * lh + 12
            list_top = y0 + self.font_big.get_linesize() + 2 * lh + 20
            footer_h = 2 * lh + 18
            visible = max(1, (panel_h - (list_top - y0) - footer_h) // row_h)
            first = max(0, min(selected - visible // 2, len(items) - visible))
            shown = items[first:first + visible]

            def fit(s, max_width):
                if f.size(s)[0] <= max_width:
                    return s
                suffix = "..."
                while s and f.size(s + suffix)[0] > max_width:
                    s = s[:-1]
                return s + suffix

            for offset, item in enumerate(shown):
                index = first + offset
                row_y = list_top + offset * row_h
                active = index == selected
                if active:
                    pg.draw.rect(surf, (32, 78, 88, 225),
                                 (x0 + 12, row_y - 4, panel_w - 24, row_h - 2))
                marker = ">" if active else " "
                current = "  [CURRENT]" if item["id"] == track_menu.get("current") else ""
                color = (255, 150, 55) if active else white
                text(fit(f"{marker} {item['name']}{current}", panel_w - 52), x0 + 24, row_y, color)
                text(fit(item["description"], panel_w - 76), x0 + 48, row_y + lh,
                     (185, 195, 205))
            if first > 0:
                text("↑", x0 + panel_w - 28, list_top, yellow, anchor="tc")
            if first + visible < len(items):
                text("↓", x0 + panel_w - 28, y0 + panel_h - footer_h - lh, yellow, anchor="tc")
            text("ENTER SELECTS   T / ESC CANCELS", self.w // 2,
                 y0 + panel_h - lh - 10, (185, 195, 205), anchor="tc")
        data = pg.image.tobytes(surf, "RGBA", True)
        self.osd_tex.write(data)
