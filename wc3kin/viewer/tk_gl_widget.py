from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import traceback
from .mesh_provider import MeshData
from .types import Pose, Rig


# Embed OpenGL in Tk via pyopengltk (Windows-friendly)
try:
    from pyopengltk import OpenGLFrame
except Exception as e:  # pragma: no cover
    OpenGLFrame = None  # type: ignore[assignment]
    _OPENGLFRAME_IMPORT_ERR = e
else:
    _OPENGLFRAME_IMPORT_ERR = None

try:
    from OpenGL.GL import (
        GL_ALPHA_TEST,
        GL_BLEND,
        GL_CLAMP_TO_EDGE,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DST_COLOR,
        GL_GREATER,
        GL_LINES,
        GL_LINEAR,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_MODELVIEW,
        GL_NICEST,
        GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_PROJECTION,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_UNSIGNED_BYTE,
        GL_ZERO,
        GL_TEXTURE_ENV_COLOR,
        GL_SRC_COLOR,
        glAlphaFunc,
        glBegin,
        glBindTexture,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glDepthMask,
        glDisable,
        glEnable,
        glEnd,
        glFlush,
        glFrustum,
        glGenTextures,
        glGetError,
        glHint,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glRotatef,
        glTexCoord2f,
        glTexEnvi,
        glTexEnvfv,
        glTexImage2D,
        glTexParameteri,
        glTranslatef,
        glVertex3f,
        glViewport,
        GL_CULL_FACE,
        GL_TEXTURE_ENV,
        GL_TEXTURE_ENV_MODE,
        GL_COMBINE,
        GL_COMBINE_RGB,
        GL_COMBINE_ALPHA,
        GL_REPLACE,
        GL_MODULATE,
        GL_SOURCE0_RGB,
        GL_OPERAND0_RGB,
        GL_SOURCE0_ALPHA,
        GL_OPERAND0_ALPHA,
        GL_SOURCE1_RGB,
        GL_OPERAND1_RGB,
        GL_SOURCE1_ALPHA,
        GL_OPERAND1_ALPHA,
        GL_TEXTURE,
        GL_CONSTANT,
        GL_BACK,
    )
    from OpenGL.error import GLError
except Exception as e:  # pragma: no cover
    _PYOPENGL_IMPORT_ERR = e
else:
    _PYOPENGL_IMPORT_ERR = None

def _iter_pose_positions(world_pos):
    """
    Accepts either:
      - dict[bone_id -> (x,y,z)]
      - list[(x,y,z)]
    Returns an iterable of (x,y,z)
    """
    if hasattr(world_pos, "values"):
        return world_pos.values()
    return world_pos

class GLViewerFrame(tk.Frame):
    """
    Wrapper frame that either hosts the real OpenGL widget (OpenGLFrame),
    or shows a helpful error message if deps are missing.

    Orbit camera (arcball):
      - RMB drag: orbit (arcball, continuous spin)
      - Shift + MMB drag: pan center
      - Wheel: dolly
      - Ctrl+R: reset camera
    """

    # -------- math helpers --------
    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = v
        n = math.sqrt(x * x + y * y + z * z) or 1.0
        return (x / n, y / n, z / n)

    @staticmethod
    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        ax, ay, az = a
        bx, by, bz = b
        return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    # -------- quaternion helpers --------
    @staticmethod
    def _quat_mul(
        q1: Tuple[float, float, float, float],
        q2: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        # (w, x, y, z)
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    @staticmethod
    def _quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, x, y, z = q
        return (w, -x, -y, -z)

    @staticmethod
    def _quat_norm(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, x, y, z = q
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        return (w / n, x / n, y / n, z / n)

    @staticmethod
    def _quat_from_axis_angle(axis: Tuple[float, float, float], angle_rad: float) -> Tuple[float, float, float, float]:
        ax, ay, az = GLViewerFrame._normalize(axis)
        s = math.sin(angle_rad * 0.5)
        return (math.cos(angle_rad * 0.5), ax * s, ay * s, az * s)

    @staticmethod
    def _quat_to_axis_angle(q: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float], float]:
        # Returns (axis, angle_deg) for glRotatef
        w, x, y, z = GLViewerFrame._quat_norm(q)
        w = GLViewerFrame._clamp(w, -1.0, 1.0)
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1e-8:
            return ((0.0, 1.0, 0.0), 0.0)
        axis = (x / s, y / s, z / s)
        return (axis, math.degrees(angle))

    @staticmethod
    def _quat_rotate_vec(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        # v' = q * (0,v) * q_conj
        qn = GLViewerFrame._quat_norm(q)
        w, x, y, z = qn
        vx, vy, vz = v

        # t = 2 * cross(q_vec, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        # v' = v + w*t + cross(q_vec, t)
        vpx = vx + w * tx + (y * tz - z * ty)
        vpy = vy + w * ty + (z * tx - x * tz)
        vpz = vz + w * tz + (x * ty - y * tx)
        return (vpx, vpy, vpz)

    # -------- arcball mapping --------
    @staticmethod
    def _arcball_point(x: int, y: int, w: int, h: int) -> Tuple[float, float, float]:
        # Map x,y in window -> point on virtual unit sphere / hyperbolic sheet
        if w <= 1 or h <= 1:
            return (0.0, 0.0, 1.0)

        nx = (2.0 * x - w) / float(w)
        ny = (h - 2.0 * y) / float(h)  # y up
        r2 = nx * nx + ny * ny
        if r2 <= 1.0:
            z = math.sqrt(1.0 - r2)
            return (nx, ny, z)

        inv_len = 1.0 / math.sqrt(r2)
        return (nx * inv_len, ny * inv_len, 0.0)

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)

        if _PYOPENGL_IMPORT_ERR is not None or OpenGLFrame is None:
            msg = "OpenGL viewer unavailable.\n\n"
            if _PYOPENGL_IMPORT_ERR is not None:
                msg += f"PyOpenGL import error: {_PYOPENGL_IMPORT_ERR!r}\n\n"
            if OpenGLFrame is None:
                msg += f"pyopengltk import error: {_OPENGLFRAME_IMPORT_ERR!r}\n\n"
            msg += "Install:\n  pip install PyOpenGL PyOpenGL_accelerate pyopengltk\n"
            tk.Label(self, text=msg, justify="left").pack(fill="both", expand=True, padx=10, pady=10)
            self._impl = None
            return

        class _Impl(OpenGLFrame):
            # camera state
            _cam_center: Tuple[float, float, float]
            _cam_rot_q: Tuple[float, float, float, float]  # (w,x,y,z)
            _cam_dist: float
            _cam_default: Dict[str, Any]
            _auto_center: bool

            # drag bookkeeping
            _drag_mode: str  # "orbit" | "pan" | ""
            _drag_last_xy: Tuple[int, int]
            _arcball_last: Tuple[float, float, float]
            _show_bones: bool

            def _tri_color(self_inner, t: int) -> tuple[float, float, float]:
                # deterministic pseudo-random color from triangle index
                x = (t * 1103515245 + 12345) & 0x7fffffff
                r = ((x >> 16) & 255) / 255.0
                g = ((x >> 8) & 255) / 255.0
                b = (x & 255) / 255.0
                # avoid super-dark colors
                r = 0.2 + 0.8 * r
                g = 0.2 + 0.8 * g
                b = 0.2 + 0.8 * b
                return (r, g, b)

            def _get_team_color_rgb(self_inner, player: int) -> tuple[float, float, float]:
                colors = [
                    (1.00, 0.05, 0.05),  # 0 red
                    (0.00, 0.26, 1.00),  # 1 blue
                    (0.10, 1.00, 0.10),  # 2 teal-ish / green
                    (0.55, 0.00, 0.78),  # 3 purple
                    (1.00, 1.00, 0.10),  # 4 yellow
                    (1.00, 0.55, 0.10),  # 5 orange
                    (0.10, 0.95, 1.00),  # 6 cyan
                    (1.00, 0.65, 0.80),  # 7 pink
                    (0.75, 0.75, 0.75),  # 8 grey
                    (0.20, 0.75, 0.25),  # 9 light green
                    (0.55, 0.35, 0.15),  # 10 brown
                    (0.05, 0.05, 0.05),  # 11 black
                ]
                if player < 0:
                    player = 0
                return colors[player % len(colors)]

            def _mat4_mul(self_inner, A, B):
                out = [[0.0] * 4 for _ in range(4)]
                for r in range(4):
                    Ar = A[r]
                    for c in range(4):
                        out[r][c] = Ar[0] * B[0][c] + Ar[1] * B[1][c] + Ar[2] * B[2][c] + Ar[3] * B[3][c]
                return out

            def _mat3_inv(self_inner, m):
                a, b, c = m[0]
                d, e, f = m[1]
                g, h, i = m[2]
                det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
                if abs(det) < 1e-12:
                    return None
                invdet = 1.0 / det
                return [
                    [(e * i - f * h) * invdet, (c * h - b * i) * invdet, (b * f - c * e) * invdet],
                    [(f * g - d * i) * invdet, (a * i - c * g) * invdet, (c * d - a * f) * invdet],
                    [(d * h - e * g) * invdet, (b * g - a * h) * invdet, (a * e - b * d) * invdet],
                ]

            def _mat4_inv_affine(self_inner, M):
                # Invert affine 4x4: [R t; 0 1]
                R = [
                    [float(M[0][0]), float(M[0][1]), float(M[0][2])],
                    [float(M[1][0]), float(M[1][1]), float(M[1][2])],
                    [float(M[2][0]), float(M[2][1]), float(M[2][2])],
                ]
                t = [float(M[0][3]), float(M[1][3]), float(M[2][3])]
                Rinv = self_inner._mat3_inv(R)
                if Rinv is None:
                    return None

                tx = -(Rinv[0][0] * t[0] + Rinv[0][1] * t[1] + Rinv[0][2] * t[2])
                ty = -(Rinv[1][0] * t[0] + Rinv[1][1] * t[1] + Rinv[1][2] * t[2])
                tz = -(Rinv[2][0] * t[0] + Rinv[2][1] * t[1] + Rinv[2][2] * t[2])

                return [
                    [Rinv[0][0], Rinv[0][1], Rinv[0][2], tx],
                    [Rinv[1][0], Rinv[1][1], Rinv[1][2], ty],
                    [Rinv[2][0], Rinv[2][1], Rinv[2][2], tz],
                    [0.0, 0.0, 0.0, 1.0],
                ]

            def _ensure_inv_bind(self_inner, current_pose):
                if getattr(self_inner, "_inv_bind_world", None) is not None:
                    return

                src_pose = self_inner._bind_pose or current_pose
                if src_pose is None:
                    self_inner._inv_bind_world = {}
                    self_inner._dbg("[skinning] no pose available for inv_bind")
                    return

                wm = getattr(src_pose, "world_mats", None) or {}
                inv = {}
                for bid, M in wm.items():
                    try:
                        bid_int = int(bid)
                    except Exception:
                        continue
                    Mi = self_inner._mat4_inv_affine(M)
                    if Mi is not None:
                        inv[bid_int] = Mi

                self_inner._inv_bind_world = inv
                self_inner._dbg(
                    f"[skinning] cached inv_bind_world for {len(inv)} bones (using {'bind_pose' if self_inner._bind_pose else 'current_pose'})"
                )

            def __init__(self_inner, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                self_inner._player_index = 0

                # debug flags
                self_inner._dbg_disable_textures = False
                self_inner._dbg_color_by_tri = False

                # IMPORTANT: when True, force opaque pipeline (no alpha test, no blending)
                self_inner._dbg_alpha_off = True

                # UV debugging (very common: V is upside-down)
                self_inner._dbg_flip_v = False

                # TeamColor debugging
                # modes: "wc3_mask", "modulate", "replace_rgb_keep_alpha", "off"
                self_inner._dbg_teamcolor_mode = "wc3_mask"
                # blend override: "layer", "alpha", "add", "none"
                self_inner._dbg_teamcolor_blend = "layer"

                # camera defaults
                self_inner._cam_center = (0.0, 0.0, 0.0)
                self_inner._cam_rot_q = (1.0, 0.0, 0.0, 0.0)
                self_inner._cam_dist = 900.0
                self_inner._cam_default = {}
                self_inner._auto_center = True

                # drag bookkeeping
                self_inner._drag_mode = ""
                self_inner._drag_last_xy = (0, 0)
                self_inner._arcball_last = (0.0, 0.0, 1.0)
                self_inner._mesh = None
                self_inner._show_bones = True

                self_inner._enabled_geosets = None

                self_inner._tex_cache = {}
                self_inner._active_tex_id = None

                self_inner._dbg_last_print = 0.0
                self_inner._dbg_every_s = 1.0
                self_inner._dbg_enabled = True

                self_inner._bind_pose = None
                self_inner._inv_bind_world = None

            # ---- hard state reset helpers (prevents "missing triangles" due to leftover GL state) ----
            def _force_solid_no_discard(self_inner) -> None:
                """Absolutely prevent alpha/texture/blend state from discarding fragments."""
                glDisable(GL_ALPHA_TEST)
                glDisable(GL_BLEND)
                glDisable(GL_TEXTURE_2D)
                glDepthMask(True)

                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                glColor4f(1.0, 1.0, 1.0, 1.0)

            def _dbg_force_opaque_state(self_inner, keep_textures: bool = True) -> None:
                """
                Force GL into a state where NOTHING can discard fragments.
                Optionally keep textures enabled so you can see UV/textures without any alpha effects.
                """
                glDisable(GL_ALPHA_TEST)
                glDisable(GL_BLEND)
                glDepthMask(True)

                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                glColor4f(1.0, 1.0, 1.0, 1.0)

                if not keep_textures:
                    glDisable(GL_TEXTURE_2D)

            def _dbg(self_inner, msg: str) -> None:
                if not getattr(self_inner, "_dbg_enabled", True):
                    return
                now = time.monotonic()
                last = float(getattr(self_inner, "_dbg_last_print", 0.0))
                every = float(getattr(self_inner, "_dbg_every_s", 1.0))
                if now - last < every:
                    return
                self_inner._dbg_last_print = now
                print(msg)

            def _gl_check(self_inner, where: str) -> None:
                err = glGetError()
                if err != 0:
                    print(f"[gl] ERROR {err} at {where}")

            def set_player_index(self_inner, idx: int) -> None:
                try:
                    idx = int(idx)
                except Exception:
                    idx = 0
                idx = max(0, min(11, idx))
                if idx == getattr(self_inner, "_player_index", 0):
                    return
                self_inner._player_index = idx
                self_inner.request_redraw()

            def get_player_index(self_inner) -> int:
                try:
                    return int(getattr(self_inner, "_player_index", 0))
                except Exception:
                    return 0

            def _resolve_texture_png(self_inner, name: str) -> str:
                import os
                base = os.path.basename(name)
                if base.lower().endswith(".blp"):
                    base = base[:-4] + ".png"
                elif not base.lower().endswith(".png"):
                    base = base + ".png"
                return r"d:\all_textures\\" + base

            def _get_texture_id(self_inner, tex_name: str) -> Optional[int]:
                if not tex_name:
                    return None

                path = self_inner._resolve_texture_png(tex_name)
                if path in self_inner._tex_cache:
                    return self_inner._tex_cache[path]

                try:
                    import os
                    if not os.path.exists(path):
                        print(f"[viewer] texture missing: {path}")
                        return None

                    img = Image.open(path).convert("RGBA")
                    w, h = img.size

                    # NOTE: leave texture as-is; use _dbg_flip_v to test UV orientation
                    data = img.tobytes("raw", "RGBA", 0, -1)

                    tid = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, tid)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

                    self_inner._tex_cache[path] = tid
                    print(f"[viewer] loaded texture {path} ({w}x{h}) -> id={tid}")
                    return tid
                except Exception as e:
                    print(f"[viewer] texture load failed: {path} err={e!r}")
                    return None

            def _get_texture_id_from_texture_entry(self_inner, entry: dict) -> Optional[int]:
                import os

                if not entry or not isinstance(entry, dict):
                    return None

                png_name = None

                if "replaceable_id" in entry:
                    rid = int(entry.get("replaceable_id", -1))
                    player = int(getattr(self_inner, "_player_index", 0))

                    if rid == 1:
                        png_name = f"TeamColor{player:02d}.png"
                    elif rid == 2:
                        png_name = f"TeamGlow{player:02d}.png"
                    else:
                        return None

                elif "image" in entry:
                    base = os.path.basename(entry["image"])
                    name, _ext = os.path.splitext(base)
                    png_name = name + ".png"

                if not png_name:
                    return None

                png_path = os.path.join(r"d:\all_textures", png_name)

                cache = getattr(self_inner, "_tex_cache", None)
                if cache is None:
                    cache = {}
                    self_inner._tex_cache = cache

                if png_path in cache:
                    return cache[png_path]

                if not os.path.exists(png_path):
                    if not getattr(self_inner, "_warned_missing_tex", False):
                        print(f"[viewer] texture not found: {png_path}")
                        self_inner._warned_missing_tex = True
                    return None

                try:
                    img = Image.open(png_path).convert("RGBA")
                    w, h = img.size
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # keep consistent with existing behavior
                    data = img.tobytes()

                    tex_id = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, tex_id)

                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

                    glTexImage2D(
                        GL_TEXTURE_2D,
                        0,
                        GL_RGBA,
                        w,
                        h,
                        0,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        data,
                    )

                    cache[png_path] = tex_id
                    print(f"[viewer] loaded texture {png_name} ({w}x{h})")
                    return tex_id

                except Exception as e:
                    print(f"[viewer] FAILED loading texture {png_path}: {e!r}")
                    return None

            def _apply_filter_mode(self_inner, filter_mode: str, alpha: float) -> tuple[bool, bool]:
                """
                Apply WC3 FilterMode to fixed-function GL state.
                Returns (blending_enabled, writes_depth).
                If _dbg_alpha_off is True, this function will NEVER enable alpha test or blending.
                """
                if getattr(self_inner, "_dbg_alpha_off", False):
                    glDisable(GL_ALPHA_TEST)
                    glDisable(GL_BLEND)
                    glDepthMask(True)
                    return (False, True)

                fm = (filter_mode or "None").lower()

                if fm in ("none",):
                    glDisable(GL_BLEND)
                    glEnable(GL_ALPHA_TEST)
                    glAlphaFunc(GL_GREATER, 0.01)
                    glDepthMask(True)
                    return (False, True)

                glDisable(GL_ALPHA_TEST)

                if fm in ("transparent", "blend"):
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glDepthMask(False)
                    return (True, False)

                if fm in ("additive", "addalpha"):
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                    glDepthMask(False)
                    return (True, False)

                if fm in ("modulate",):
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_DST_COLOR, GL_ZERO)
                    glDepthMask(False)
                    return (True, False)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(False)
                return (True, False)

            def set_geosets_enabled(self_inner, enabled: Optional[list[bool]]) -> None:
                self_inner._enabled_geosets = enabled
                self_inner.request_redraw()

            def get_geosets_enabled(self_inner) -> Optional[list[bool]]:
                return getattr(self_inner, "_enabled_geosets", None)

            def set_enabled_geosets(self_inner, enabled: Optional[list[bool]]) -> None:
                self_inner.set_geosets_enabled(enabled)

            def get_enabled_geosets(self_inner) -> Optional[list[bool]]:
                return self_inner.get_geosets_enabled()

            def set_show_bones(self_inner, show: bool) -> None:
                show = bool(show)
                if show == getattr(self_inner, "_show_bones", True):
                    return
                self_inner._show_bones = show
                self_inner.request_redraw()

            def get_show_bones(self_inner) -> bool:
                return bool(getattr(self_inner, "_show_bones", True))

            def initgl(self_inner) -> None:
                glClearColor(0.08, 0.08, 0.10, 1.0)
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glDisable(GL_DEPTH_TEST)

            def request_redraw(self_inner) -> None:
                if hasattr(self_inner, "_display"):
                    self_inner.after_idle(self_inner._display)  # type: ignore[attr-defined]
                elif hasattr(self_inner, "tkRedraw"):
                    self_inner.after_idle(self_inner.tkRedraw)  # type: ignore[attr-defined]
                else:
                    self_inner.after_idle(self_inner.redraw)

            # -------- camera API --------
            def get_camera_state(self_inner) -> Dict[str, Any]:
                cx, cy, cz = self_inner._cam_center
                qw, qx, qy, qz = GLViewerFrame._quat_norm(self_inner._cam_rot_q)
                return {
                    "center": [float(cx), float(cy), float(cz)],
                    "dist": float(self_inner._cam_dist),
                    "rot_q": [float(qw), float(qx), float(qy), float(qz)],
                }

            def set_camera_state(self_inner, state: Dict[str, Any]) -> None:
                c = state.get("center") or [0.0, 0.0, 0.0]
                if isinstance(c, (list, tuple)) and len(c) >= 3:
                    self_inner._cam_center = (float(c[0]), float(c[1]), float(c[2]))

                if "dist" in state:
                    self_inner._cam_dist = max(5.0, float(state["dist"]))

                q = state.get("rot_q")
                if isinstance(q, (list, tuple)) and len(q) >= 4:
                    qw, qx, qy, qz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    self_inner._cam_rot_q = GLViewerFrame._quat_norm((qw, qx, qy, qz))
                else:
                    yaw = state.get("yaw_deg")
                    pitch = state.get("pitch_deg")
                    if isinstance(yaw, (int, float)) and isinstance(pitch, (int, float)):
                        q_yaw = GLViewerFrame._quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(float(yaw)))
                        q_pitch = GLViewerFrame._quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(float(pitch)))
                        self_inner._cam_rot_q = GLViewerFrame._quat_norm(GLViewerFrame._quat_mul(q_yaw, q_pitch))

                self_inner._auto_center = False

            def snapshot_default_camera(self_inner) -> None:
                if not self_inner._cam_default:
                    self_inner._cam_default = self_inner.get_camera_state()

            def reset_camera(self_inner) -> None:
                if self_inner._cam_default:
                    self_inner.set_camera_state(dict(self_inner._cam_default))
                    self_inner._auto_center = True
                else:
                    self_inner._cam_center = (0.0, 0.0, 0.0)
                    self_inner._cam_rot_q = (1.0, 0.0, 0.0, 0.0)
                    self_inner._cam_dist = 900.0
                    self_inner._auto_center = True
                self_inner.request_redraw()

            def fit_camera_to_pose(self_inner, pose: Pose) -> None:
                pos_iter = _iter_pose_positions(pose.world_pos)
                xs = [p[0] for p in pos_iter if p is not None]
                ys = [p[1] for p in pos_iter if p is not None]
                zs = [p[2] for p in pos_iter if p is not None]
                if not xs or not ys or not zs:
                    return

                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                minz, maxz = min(zs), max(zs)
                cx = (minx + maxx) * 0.5
                cy = (miny + maxy) * 0.5
                cz = (minz + maxz) * 0.5
                span = max(maxx - minx, maxy - miny, maxz - minz)

                self_inner._cam_center = (float(cx), float(cy), float(cz))
                self_inner._cam_dist = max(50.0, float(span) * 1.8)

                q_yaw = GLViewerFrame._quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(35.0))
                q_pitch = GLViewerFrame._quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(-20.0))
                self_inner._cam_rot_q = GLViewerFrame._quat_norm(GLViewerFrame._quat_mul(q_yaw, q_pitch))

                self_inner._auto_center = True
                self_inner.request_redraw()

            # -------- input handling --------
            def _begin_orbit(self_inner, e: tk.Event) -> None:
                self_inner._drag_mode = "orbit"
                self_inner._drag_last_xy = (int(e.x), int(e.y))
                self_inner._auto_center = False

                w = max(1, int(self_inner.winfo_width()))
                h = max(1, int(self_inner.winfo_height()))
                self_inner._arcball_last = GLViewerFrame._arcball_point(int(e.x), int(e.y), w, h)

            def _begin_pan(self_inner, e: tk.Event) -> None:
                self_inner._drag_mode = "pan"
                self_inner._drag_last_xy = (int(e.x), int(e.y))
                self_inner._auto_center = False

            def _end_drag(self_inner, _e: tk.Event) -> None:
                self_inner._drag_mode = ""

            def _on_drag(self_inner, e: tk.Event) -> None:
                mode = self_inner._drag_mode
                if not mode:
                    return
                x, y = int(e.x), int(e.y)
                lx, ly = self_inner._drag_last_xy
                dx = x - lx
                dy = y - ly
                self_inner._drag_last_xy = (x, y)

                if mode == "orbit":
                    w = max(1, int(self_inner.winfo_width()))
                    h = max(1, int(self_inner.winfo_height()))
                    p0 = self_inner._arcball_last
                    p1 = GLViewerFrame._arcball_point(x, y, w, h)
                    self_inner._arcball_last = p1

                    axis = GLViewerFrame._cross(p0, p1)
                    axis_len = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
                    if axis_len > 1e-8:
                        dot = GLViewerFrame._dot(p0, p1)
                        dot = GLViewerFrame._clamp(dot, -1.0, 1.0)
                        angle = math.acos(dot)
                        angle = -angle

                        dq = GLViewerFrame._quat_from_axis_angle(axis, angle)
                        self_inner._cam_rot_q = GLViewerFrame._quat_norm(GLViewerFrame._quat_mul(dq, self_inner._cam_rot_q))
                        self_inner.request_redraw()
                    return

                if mode == "pan":
                    dist = float(self_inner._cam_dist)
                    pan_scale = dist * 0.0025

                    q = GLViewerFrame._quat_conj(self_inner._cam_rot_q)
                    right = GLViewerFrame._quat_rotate_vec(q, (1.0, 0.0, 0.0))
                    up = GLViewerFrame._quat_rotate_vec(q, (0.0, 1.0, 0.0))

                    cx, cy, cz = self_inner._cam_center
                    cx -= right[0] * dx * pan_scale
                    cy -= right[1] * dx * pan_scale
                    cz -= right[2] * dx * pan_scale
                    cx += up[0] * dy * pan_scale
                    cy += up[1] * dy * pan_scale
                    cz += up[2] * dy * pan_scale
                    self_inner._cam_center = (cx, cy, cz)
                    self_inner.request_redraw()

            def _on_wheel(self_inner, e: tk.Event) -> None:
                delta = getattr(e, "delta", 0) or 0
                step = 0.10
                if delta > 0:
                    self_inner._cam_dist *= (1.0 - step)
                elif delta < 0:
                    self_inner._cam_dist *= (1.0 + step)
                self_inner._cam_dist = max(5.0, float(self_inner._cam_dist))
                self_inner._auto_center = False
                self_inner.request_redraw()

            def _on_reset_key(self_inner, _e: tk.Event) -> None:
                self_inner.reset_camera()

            def _apply_teamcolor(self_inner, team_rgb: tuple[float, float, float], alpha: float) -> None:
                """
                Apply teamcolor combine based on debug mode.
                Assumes a teamcolor texture is bound.
                """
                mode = str(getattr(self_inner, "_dbg_teamcolor_mode", "wc3_mask") or "wc3_mask").lower()

                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE)

                # Common: constant color = team RGB, constant alpha = layer alpha
                glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (team_rgb[0], team_rgb[1], team_rgb[2], float(alpha)))

                if mode == "off":
                    # behave like normal textured layer
                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                    glColor4f(1.0, 1.0, 1.0, float(alpha))
                    return

                if mode == "modulate":
                    # RGB = texture.rgb * constant.rgb
                    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_MODULATE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_TEXTURE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_RGB, GL_SRC_COLOR)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_CONSTANT)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND1_RGB, GL_SRC_COLOR)

                    # Alpha = texture.alpha * constant.alpha
                    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, GL_MODULATE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_ALPHA, GL_TEXTURE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_ALPHA, GL_SRC_ALPHA)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_ALPHA, GL_CONSTANT)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND1_ALPHA, GL_SRC_ALPHA)

                    glColor4f(1.0, 1.0, 1.0, 1.0)
                    return

                if mode == "replace_rgb_keep_alpha":
                    # RGB = constant.rgb, Alpha = texture.alpha * constant.alpha
                    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_REPLACE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_CONSTANT)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_RGB, GL_SRC_COLOR)

                    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, GL_MODULATE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_ALPHA, GL_TEXTURE)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_ALPHA, GL_SRC_ALPHA)
                    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_ALPHA, GL_CONSTANT)
                    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND1_ALPHA, GL_SRC_ALPHA)

                    glColor4f(1.0, 1.0, 1.0, 1.0)
                    return

                # default: "wc3_mask" (your current)
                # RGB = constant.rgb, Alpha = texture.alpha * constant.alpha
                glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_REPLACE)
                glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_CONSTANT)
                glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_RGB, GL_SRC_COLOR)

                glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, GL_MODULATE)
                glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_ALPHA, GL_TEXTURE)
                glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_ALPHA, GL_SRC_ALPHA)
                glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_ALPHA, GL_CONSTANT)
                glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND1_ALPHA, GL_SRC_ALPHA)

                glColor4f(1.0, 1.0, 1.0, 1.0)

            def _apply_teamcolor_blend(self_inner, filter_mode: str, alpha: float) -> None:
                """
                Apply blending for teamcolor layers based on debug override.
                """
                if getattr(self_inner, "_dbg_alpha_off", False):
                    glDisable(GL_ALPHA_TEST)
                    glDisable(GL_BLEND)
                    glDepthMask(True)
                    return

                mode = str(getattr(self_inner, "_dbg_teamcolor_blend", "layer") or "layer").lower()
                if mode == "none":
                    glDisable(GL_ALPHA_TEST)
                    glDisable(GL_BLEND)
                    glDepthMask(True)
                    return

                if mode == "alpha":
                    glDisable(GL_ALPHA_TEST)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glDepthMask(False)
                    return

                if mode == "add":
                    glDisable(GL_ALPHA_TEST)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                    glDepthMask(False)
                    return

                # default: layer-driven
                self_inner._apply_filter_mode(filter_mode, alpha)

            def redraw(self_inner) -> None:
                w = int(self_inner.winfo_width())
                h = int(self_inner.winfo_height())
                if w <= 1 or h <= 1:
                    return
                glDisable(GL_CULL_FACE)
                glViewport(0, 0, w, h)

                # Projection
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect = float(w) / float(h)

                fov_deg = 45.0
                z_near = 5.0
                z_far = 200000.0
                top = math.tan(math.radians(fov_deg * 0.5)) * z_near
                right = top * aspect
                glFrustum(-right, right, -top, top, z_near, z_far)

                # ModelView
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                pose = getattr(self_inner, "_pose", None)
                rig = getattr(self_inner, "_rig", None)
                mesh = getattr(self_inner, "_mesh", None)

                try:
                    self_inner._ensure_inv_bind(pose)
                except Exception as e:
                    self_inner._dbg(f"[skinning] ensure_inv_bind failed: {e!r}")

                if pose is None or rig is None:
                    glFlush()
                    return

                # One-shot auto-center
                if getattr(self_inner, "_auto_center", False):
                    pos_iter = _iter_pose_positions(pose.world_pos)
                    xs = [p[0] for p in pos_iter if p is not None]
                    ys = [p[1] for p in pos_iter if p is not None]
                    zs = [p[2] for p in pos_iter if p is not None]
                    if xs and ys and zs:
                        self_inner._cam_center = (
                            (min(xs) + max(xs)) * 0.5,
                            (min(ys) + max(ys)) * 0.5,
                            (min(zs) + max(zs)) * 0.5,
                        )
                    self_inner._auto_center = False

                # Camera transform
                cx, cy, cz = self_inner._cam_center
                dist = float(self_inner._cam_dist)

                glTranslatef(0.0, 0.0, -dist)
                axis, angle_deg = GLViewerFrame._quat_to_axis_angle(self_inner._cam_rot_q)
                if angle_deg != 0.0:
                    glRotatef(-float(angle_deg), float(axis[0]), float(axis[1]), float(axis[2]))
                glTranslatef(-float(cx), -float(cy), -float(cz))

                # Debug alpha off: allow textures, forbid any discard
                if getattr(self_inner, "_dbg_alpha_off", False):
                    self_inner._dbg_force_opaque_state(keep_textures=True)

                mesh = getattr(self_inner, "_mesh", None)

                meshes_to_draw = []
                if mesh is not None and getattr(mesh, "submeshes", None):
                    meshes_to_draw = list(getattr(mesh, "submeshes") or [])
                elif mesh is not None:
                    meshes_to_draw = [mesh]

                enabled = getattr(self_inner, "_enabled_geosets", None)
                if enabled and len(enabled) == len(meshes_to_draw):
                    meshes_to_draw = [m for m, ok in zip(meshes_to_draw, enabled) if ok]

                for submesh in meshes_to_draw:
                    if submesh is None:
                        continue

                    try:
                        from .evaluator import transform_point

                        verts = submesh.vertices
                        tris = submesh.triangles
                        uvs = getattr(submesh, "uvs", None)

                        vgroups = getattr(submesh, "vertex_groups", None)
                        gmat = getattr(submesh, "groups_matrices", None)

                        def skin_vertex(v, vid):
                            if vgroups is None or gmat is None or pose is None:
                                return v
                            if vid < 0 or vid >= len(vgroups):
                                return v

                            gid = vgroups[vid]
                            if gid is None or gid < 0 or gid >= len(gmat):
                                return v

                            bids = gmat[gid]
                            if not bids:
                                return v

                            inv_bind = getattr(self_inner, "_inv_bind_world", None) or {}
                            world_mats = getattr(pose, "world_mats", None) or {}

                            n = 0
                            accx = accy = accz = 0.0
                            for bid in bids:
                                mtx = world_mats.get(bid) or world_mats.get(str(bid))
                                invb = inv_bind.get(bid)
                                if invb is None:
                                    try:
                                        invb = inv_bind.get(int(bid))
                                    except Exception:
                                        invb = None

                                if mtx is None or invb is None:
                                    continue

                                skin_mtx = self_inner._mat4_mul(mtx, invb)
                                tv = transform_point(skin_mtx, v)

                                accx += tv[0]
                                accy += tv[1]
                                accz += tv[2]
                                n += 1

                            if n <= 0:
                                return v

                            invn = 1.0 / n
                            return (accx * invn, accy * invn, accz * invn)

                        materials = getattr(submesh, "materials", None)
                        textures = getattr(submesh, "textures", None)
                        mid = getattr(submesh, "geoset_material_id", None)

                        layers = None
                        if (
                            uvs is not None
                            and materials is not None
                            and textures is not None
                            and mid is not None
                            and 0 <= int(mid) < len(materials)
                        ):
                            layers = materials[int(mid)].get("layers") or None

                        # ---- SOLID/DEBUG PATH ----
                        if getattr(self_inner, "_dbg_disable_textures", False) or getattr(self_inner, "_dbg_color_by_tri", False):
                            layers = None

                        if not layers:
                            # CRITICAL FIX:
                            # Always hard reset state so triangles cannot vanish due to leftover alpha/tex state.
                            self_inner._force_solid_no_discard()

                            glColor3f(0.35, 0.7, 0.35)

                            vlen = len(verts)
                            began = False
                            t_index = 0
                            try:
                                glBegin(GL_TRIANGLES)
                                began = True
                                for (i0, i1, i2) in tris:
                                    if getattr(self_inner, "_dbg_color_by_tri", False):
                                        r, g, b = self_inner._tri_color(t_index)
                                        glColor3f(r, g, b)
                                    t_index += 1

                                    if (
                                        i0 < 0 or i1 < 0 or i2 < 0
                                        or i0 >= vlen or i1 >= vlen or i2 >= vlen
                                    ):
                                        continue

                                    for vid in (i0, i1, i2):
                                        v = skin_vertex(verts[vid], vid)
                                        glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                            finally:
                                if began:
                                    glEnd()

                        # ---- LAYERED/TEXTURED PATH ----
                        else:
                            glEnable(GL_DEPTH_TEST)

                            for layer in layers:
                                layer_tex_id = layer.get("texture_id", None)
                                filter_mode = layer.get("filter_mode", "None")
                                alpha = float(layer.get("alpha", 1.0) or 1.0)

                                # Bind texture for this layer
                                glDisable(GL_TEXTURE_2D)
                                tid = None
                                tex_entry = None
                                if textures is not None and layer_tex_id is not None and 0 <= int(layer_tex_id) < len(textures):
                                    tex_entry = textures[int(layer_tex_id)]
                                    tid = self_inner._get_texture_id_from_texture_entry(tex_entry)

                                if tid is not None and not getattr(self_inner, "_dbg_disable_textures", False):
                                    glEnable(GL_TEXTURE_2D)
                                    glBindTexture(GL_TEXTURE_2D, tid)

                                # Identify teamcolor
                                is_teamcolor = False
                                team_rgb = (1.0, 1.0, 1.0)
                                if tex_entry is not None:
                                    rid = tex_entry.get("replaceable_id", None)
                                    if rid is not None and int(rid) == 1:
                                        is_teamcolor = True
                                        player = int(getattr(self_inner, "_player_index", 0))
                                        team_rgb = self_inner._get_team_color_rgb(player)

                                # Decide blend state
                                if is_teamcolor and tid is not None and not getattr(self_inner, "_dbg_disable_textures", False):
                                    self_inner._apply_teamcolor_blend(filter_mode, alpha)
                                    self_inner._apply_teamcolor(team_rgb, alpha)
                                else:
                                    # Non-teamcolor: use normal filter mode, unless dbg alpha off
                                    self_inner._apply_filter_mode(filter_mode, alpha)
                                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                                    if getattr(self_inner, "_dbg_alpha_off", False):
                                        glColor4f(1.0, 1.0, 1.0, 1.0)
                                    else:
                                        glColor4f(1.0, 1.0, 1.0, float(alpha))

                                vlen = len(verts)
                                began = False
                                try:
                                    glBegin(GL_TRIANGLES)
                                    began = True
                                    for (i0, i1, i2) in tris:
                                        if (
                                            i0 < 0 or i1 < 0 or i2 < 0
                                            or i0 >= vlen or i1 >= vlen or i2 >= vlen
                                        ):
                                            continue
                                        for vid in (i0, i1, i2):
                                            v = skin_vertex(verts[vid], vid)
                                            if uvs is not None and vid < len(uvs):
                                                u, vv = uvs[vid]
                                                if getattr(self_inner, "_dbg_flip_v", False):
                                                    vv = 1.0 - float(vv)
                                                glTexCoord2f(float(u), float(vv))
                                            glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                                finally:
                                    if began:
                                        glEnd()

                                # restore to sane defaults between layers
                                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

                            # restore defaults after mesh
                            glDisable(GL_TEXTURE_2D)
                            glDisable(GL_BLEND)
                            glDisable(GL_ALPHA_TEST)
                            glDepthMask(True)

                    except GLError as ge:
                        print(f"[viewer] GL ERROR during mesh draw: {ge!r}")
                    except Exception as e:
                        print(f"[viewer] EXCEPTION during mesh draw: {e!r}")
                        traceback.print_exc()

                # --- BONES ---
                if getattr(self_inner, "_show_bones", True):
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    for oid, b in rig.bones.items():
                        pid = b.parent_id
                        if pid is None or pid not in rig.bones:
                            continue
                        p0 = pose.world_pos.get(pid)
                        p1 = pose.world_pos.get(oid)
                        if p0 is None or p1 is None:
                            continue

                        glColor3f(0.7, 0.7, 0.9)
                        glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                        glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                    glEnd()

                glFlush()

        self._impl = _Impl(self, width=640, height=480)
        self._impl.pack(fill="both", expand=True)

        self._impl.animate = 0

        self._impl._pose = None
        self._impl._rig = None
        self._impl._active_ids = None

        # mouse bindings (orbit/pan/zoom)
        self._impl.bind("<ButtonPress-3>", self._impl._begin_orbit)
        self._impl.bind("<B3-Motion>", self._impl._on_drag)
        self._impl.bind("<ButtonRelease-3>", self._impl._end_drag)

        # Shift + MMB pan
        self._impl.bind("<Shift-ButtonPress-2>", self._impl._begin_pan)
        self._impl.bind("<Shift-B2-Motion>", self._impl._on_drag)
        self._impl.bind("<ButtonRelease-2>", self._impl._end_drag)

        # wheel
        self._impl.bind("<MouseWheel>", self._impl._on_wheel)
        self._impl.bind("<Button-4>", lambda _e: self._impl._on_wheel(type("E", (), {"delta": 120})()))
        self._impl.bind("<Button-5>", lambda _e: self._impl._on_wheel(type("E", (), {"delta": -120})()))

        # reset
        self._impl.bind("<Control-r>", self._impl._on_reset_key)
        self._impl.bind("<Control-R>", self._impl._on_reset_key)

    # ---- public camera helpers ----
    def get_camera_state(self) -> Optional[Dict[str, Any]]:
        if self._impl is None:
            return None
        return self._impl.get_camera_state()

    def set_camera_state(self, state: Dict[str, Any]) -> None:
        if self._impl is None:
            return
        self._impl.set_camera_state(state)
        self._impl.request_redraw()

    def reset_camera(self) -> None:
        if self._impl is None:
            return
        self._impl.reset_camera()

    def snapshot_default_camera(self) -> None:
        if self._impl is None:
            return
        self._impl.snapshot_default_camera()

    def fit_camera_to_pose(self, pose: Pose) -> None:
        if self._impl is None:
            return
        self._impl.fit_camera_to_pose(pose)

    def set_mesh(self, mesh: MeshData) -> None:
        if self._impl is None:
            return
        self._impl._mesh = mesh

    def set_enabled_geosets(self, enabled: Optional[list[bool]]) -> None:
        if self._impl is None:
            return
        self._impl.set_enabled_geosets(enabled)

    def get_enabled_geosets(self) -> Optional[list[bool]]:
        if self._impl is None:
            return None
        return self._impl.get_enabled_geosets()

    def set_show_bones(self, show: bool) -> None:
        if self._impl is None:
            return
        self._impl.set_show_bones(show)

    def get_show_bones(self) -> bool:
        if self._impl is None:
            return True
        return self._impl.get_show_bones()

    def set_player_index(self, idx: int) -> None:
        if self._impl is None:
            return
        self._impl.set_player_index(idx)

    def get_player_index(self) -> int:
        if self._impl is None:
            return 0
        return self._impl.get_player_index()

    def set_bind_pose(self, pose: Pose) -> None:
        """Provide bind/rest pose for inverse-bind skinning."""
        if self._impl is None:
            return
        self._impl._bind_pose = pose
        self._impl._inv_bind_world = None

    def set_pose(self, pose: Pose, rig: Rig, active_ids: Optional[set[int]] = None) -> None:
        if self._impl is None:
            return
        self._impl._pose = pose
        self._impl._rig = rig
        self._impl._active_ids = active_ids
        self._impl.request_redraw()

    # ---- debug controls ----
    def set_debug_alpha_off(self, val: bool) -> None:
        if self._impl is None:
            return
        self._impl._dbg_alpha_off = bool(val)
        self._impl.request_redraw()

    def get_debug_alpha_off(self) -> bool:
        if self._impl is None:
            return False
        return bool(getattr(self._impl, "_dbg_alpha_off", False))

    def set_debug_disable_textures(self, val: bool) -> None:
        if self._impl is None:
            return
        self._impl._dbg_disable_textures = bool(val)
        self._impl.request_redraw()

    def get_debug_disable_textures(self) -> bool:
        if self._impl is None:
            return False
        return bool(getattr(self._impl, "_dbg_disable_textures", False))

    def set_debug_color_by_tri(self, val: bool) -> None:
        if self._impl is None:
            return
        self._impl._dbg_color_by_tri = bool(val)
        self._impl.request_redraw()

    def get_debug_color_by_tri(self) -> bool:
        if self._impl is None:
            return False
        return bool(getattr(self._impl, "_dbg_color_by_tri", False))

    def set_debug_enabled(self, val: bool) -> None:
        if self._impl is None:
            return
        self._impl._dbg_enabled = bool(val)
        self._impl.request_redraw()

    def get_debug_enabled(self) -> bool:
        if self._impl is None:
            return False
        return bool(getattr(self._impl, "_dbg_enabled", False))

    # ---- UV debug ----
    def set_debug_flip_v(self, val: bool) -> None:
        if self._impl is None:
            return
        self._impl._dbg_flip_v = bool(val)
        self._impl.request_redraw()

    def get_debug_flip_v(self) -> bool:
        if self._impl is None:
            return False
        return bool(getattr(self._impl, "_dbg_flip_v", False))

    # ---- teamcolor debug ----
    def set_teamcolor_mode(self, mode: str) -> None:
        if self._impl is None:
            return
        mode = str(mode or "").strip().lower()
        if mode not in ("wc3_mask", "modulate", "replace_rgb_keep_alpha", "off"):
            mode = "wc3_mask"
        self._impl._dbg_teamcolor_mode = mode
        self._impl.request_redraw()

    def get_teamcolor_mode(self) -> str:
        if self._impl is None:
            return "wc3_mask"
        return str(getattr(self._impl, "_dbg_teamcolor_mode", "wc3_mask"))

    def set_teamcolor_blend(self, mode: str) -> None:
        if self._impl is None:
            return
        mode = str(mode or "").strip().lower()
        if mode not in ("layer", "alpha", "add", "none"):
            mode = "layer"
        self._impl._dbg_teamcolor_blend = mode
        self._impl.request_redraw()

    def get_teamcolor_blend(self) -> str:
        if self._impl is None:
            return "layer"
        return str(getattr(self._impl, "_dbg_teamcolor_blend", "layer"))
