from __future__ import annotations

import math
import tkinter as tk
from typing import Optional, Dict, Any, Tuple

from .types import Pose, Rig

# Embed OpenGL in Tk via pyopengltk (Windows-friendly)
try:
    from pyopengltk import OpenGLFrame
except Exception as e:  # pragma: no cover
    OpenGLFrame = None  # type: ignore[assignment]
    _OPENGLFRAME_IMPORT_ERR = e
else:
    _OPENGLFRAME_IMPORT_ERR = None

import time
from .mesh_provider import MeshData
from PIL import Image

try:
    from OpenGL.GL import (
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,  # kept imported (not used by default path)
        glFrustum,
        glVertex3f,
        glViewport,
        glEnable,
        glHint,
        glDisable,
        glFlush,
        glTranslatef,
        glRotatef,
        glTexCoord2f,
        glBindTexture,
        glBlendFunc,
        glAlphaFunc,
        glDepthMask,
        glGenTextures,
        glTexImage2D,
        glTexParameteri,
        GL_BLEND,
        GL_ALPHA_TEST,
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_ONE,
        GL_ZERO,
        GL_DST_COLOR,
        GL_TEXTURE_2D,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        GL_LINEAR,
        GL_CLAMP_TO_EDGE,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TRIANGLES,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_LINES,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_DEPTH_TEST,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_NICEST,
        GL_GREATER,
        glGetError,
        GL_NO_ERROR,
        glTexCoord2f,
        )

except Exception as e:  # pragma: no cover
    _PYOPENGL_IMPORT_ERR = e
else:
    _PYOPENGL_IMPORT_ERR = None


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

            def __init__(self_inner, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

                # for rendering the unit completely correctly
                self_inner._player_index = 0

                # camera defaults (safe even before GL init)
                self_inner._cam_center = (0.0, 0.0, 0.0)
                self_inner._cam_rot_q = (1.0, 0.0, 0.0, 0.0)  # identity
                self_inner._cam_dist = 900.0
                self_inner._cam_default = {}
                self_inner._auto_center = True

                # drag bookkeeping
                self_inner._drag_mode = ""
                self_inner._drag_last_xy = (0, 0)
                self_inner._arcball_last = (0.0, 0.0, 1.0)
                self_inner._mesh = None
                self_inner._show_bones = True

                # which geosets/submeshes are enabled (None => all)
                self_inner._enabled_geosets = None

                # texture info
                self_inner._tex_cache = {}   # key: resolved png path -> texture id
                self_inner._active_tex_id = None

                #debug flags
                self_inner._dbg_last_print = 0.0
                self_inner._dbg_every_s = 1.0  # print at most once per second
                self_inner._dbg_enabled = True  # flip to False when you’re done

            def _get_texture_id_from_texture_entry(self_inner, entry: dict) -> Optional[int]:
                """
                Resolve a WC3 texture entry into an OpenGL texture ID.

                entry formats:
                  {"image": "Textures\\Units\\Foo.blp"}
                  {"replaceable_id": 1}  # TeamColor
                  {"replaceable_id": 2}  # TeamGlow
                """
                import os

                if not entry or not isinstance(entry, dict):
                    return None

                # ---- determine PNG filename ----
                png_name = None

                # Replaceable textures (team color / glow)
                if "replaceable_id" in entry:
                    rid = int(entry.get("replaceable_id", -1))
                    player = int(getattr(self_inner, "_player_index", 0))

                    if rid == 1:
                        png_name = f"TeamColor{player:02d}.png"
                    elif rid == 2:
                        png_name = f"TeamGlow{player:02d}.png"
                    else:
                        # Unknown replaceable — WC3 has more, but ignore for now
                        return None

                # Normal bitmap image
                elif "image" in entry:
                    base = os.path.basename(entry["image"])
                    name, _ext = os.path.splitext(base)
                    png_name = name + ".png"

                if not png_name:
                    return None

                # ---- absolute path (your hard rule) ----
                png_path = os.path.join(r"d:\all_textures", png_name)

                # ---- texture cache ----
                cache = getattr(self_inner, "_tex_cache", None)
                if cache is None:
                    cache = {}
                    self_inner._tex_cache = cache

                if png_path in cache:
                    return cache[png_path]

                # ---- load PNG & upload to OpenGL ----
                if not os.path.exists(png_path):
                    if not getattr(self_inner, "_warned_missing_tex", False):
                        print(f"[viewer] texture not found: {png_path}")
                        self_inner._warned_missing_tex = True
                    return None

                try:
                    img = Image.open(png_path).convert("RGBA")
                    w, h = img.size

                    # Flip vertically: WC3 UVs expect this
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

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

                Key behavior:
                - FilterMode "None" is treated as an opaque pass, but we *alpha-test* to discard near-zero
                texels. This prevents team-color / glow textures from painting as a flat grey overlay.
                - Blended passes keep depth-test but disable depth-write (glDepthMask(False)).
                """
                fm = (filter_mode or "None").lower()

                if fm in ("none",):
                    glDisable(GL_BLEND)
                    glEnable(GL_ALPHA_TEST)
                    glAlphaFunc(GL_GREATER, 0.01)
                    glDepthMask(True)
                    return (False, True)

                # blended passes
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
                    glBlendFunc(GL_DST_COLOR, GL_ZERO)  # multiply
                    glDepthMask(False)
                    return (True, False)

                # fallback
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(False)
                return (True, False)


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
                if err != GL_NO_ERROR:
                    # don’t throttle errors; they’re important
                    print(f"[gl] ERROR {err} at {where}")

            def set_player_index(self_inner, idx: int) -> None:
                try:
                    idx = int(idx)
                except Exception:
                    idx = 0
                if idx < 0:
                    idx = 0
                if idx > 11:
                    idx = 11
                if idx == getattr(self_inner, "_player_index", 0):
                    return
                self_inner._player_index = idx
                # if you cache textures, you may want to invalidate team textures here
                self_inner.request_redraw()

            def get_player_index(self_inner) -> int:
                try:
                    return int(getattr(self_inner, "_player_index", 0))
                except Exception:
                    return 0

            def _resolve_texture_png(self_inner, name: str) -> str:
                # basename only, replace .blp -> .png, always from d:\all_textures
                import os
                base = os.path.basename(name)
                if base.lower().endswith(".blp"):
                    base = base[:-4] + ".png"
                elif not base.lower().endswith(".png"):
                    base = base + ".png"
                return r"d:\all_textures\\" + base

            def _get_texture_id(self_inner, tex_name: str) -> Optional[int]:
                # Lazy-load and cache
                if not tex_name:
                    return None

                path = self_inner._resolve_texture_png(tex_name)
                if path in self_inner._tex_cache:
                    return self_inner._tex_cache[path]

                try:
                    from PIL import Image
                    import os

                    if not os.path.exists(path):
                        print(f"[viewer] texture missing: {path}")
                        return None

                    img = Image.open(path).convert("RGBA")
                    w, h = img.size
                    data = img.tobytes("raw", "RGBA", 0, -1)  # flip vertically so V works naturally in OpenGL

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

            def set_geosets_enabled(self_inner, enabled: Optional[list[bool]]) -> None:
                self_inner._enabled_geosets = enabled
                self_inner.request_redraw()

            def get_geosets_enabled(self_inner) -> Optional[list[bool]]:
                return getattr(self_inner, "_enabled_geosets", None)

            # Preferred names (match window wiring)
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

            # -------- camera API (new persistence format) --------
            def get_camera_state(self_inner) -> Dict[str, Any]:
                cx, cy, cz = self_inner._cam_center
                qw, qx, qy, qz = GLViewerFrame._quat_norm(self_inner._cam_rot_q)
                return {
                    "center": [float(cx), float(cy), float(cz)],
                    "dist": float(self_inner._cam_dist),
                    "rot_q": [float(qw), float(qx), float(qy), float(qz)],
                }

            def set_camera_state(self_inner, state: Dict[str, Any]) -> None:
                # Accept new format first. If old yaw/pitch exists, convert once.
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
                    # Back-compat: old keys -> convert, but we will only SAVE rot_q later.
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
                    # "like we never touched it"
                    self_inner._auto_center = True
                else:
                    self_inner._cam_center = (0.0, 0.0, 0.0)
                    self_inner._cam_rot_q = (1.0, 0.0, 0.0, 0.0)
                    self_inner._cam_dist = 900.0
                    self_inner._auto_center = True
                self_inner.request_redraw()

            def fit_camera_to_pose(self_inner, pose: Pose) -> None:
                xs = [p[0] for p in pose.world_pos.values()]
                ys = [p[1] for p in pose.world_pos.values()]
                zs = [p[2] for p in pose.world_pos.values()]
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

                # nice default orientation (3/4 view)
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

                        # Invert feel (matches your earlier inverted X/Y preference)
                        angle = -angle

                        dq = GLViewerFrame._quat_from_axis_angle(axis, angle)
                        self_inner._cam_rot_q = GLViewerFrame._quat_norm(
                            GLViewerFrame._quat_mul(dq, self_inner._cam_rot_q)
                        )
                        self_inner.request_redraw()
                    return

                if mode == "pan":
                    dist = float(self_inner._cam_dist)
                    pan_scale = dist * 0.0025

                    # camera right/up in world from quaternion orientation
                    q = GLViewerFrame._quat_conj(self_inner._cam_rot_q)
                    right = GLViewerFrame._quat_rotate_vec(q, (1.0, 0.0, 0.0))
                    up = GLViewerFrame._quat_rotate_vec(q, (0.0, 1.0, 0.0))

                    cx, cy, cz = self_inner._cam_center
                    # inverted screen mapping
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

            def redraw(self_inner) -> None:
                w = int(self_inner.winfo_width())
                h = int(self_inner.winfo_height())
                if w <= 1 or h <= 1:
                    return

                glViewport(0, 0, w, h)

                # ---- Projection ----
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect = float(w) / float(h)

                fov_deg = 45.0
                z_near = 5.0
                z_far = 200000.0
                top = math.tan(math.radians(fov_deg * 0.5)) * z_near
                right = top * aspect
                glFrustum(-right, right, -top, top, z_near, z_far)

                # ---- ModelView ----
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                self_inner._gl_check("after glClear")

                pose = getattr(self_inner, "_pose", None)
                rig = getattr(self_inner, "_rig", None)
                mesh = getattr(self_inner, "_mesh", None)

                # lightweight state dump (throttled)
                try:
                    pose_ct = len(pose.world_pos) if pose is not None else 0
                except Exception:
                    pose_ct = -1

                try:
                    rig_ct = len(rig.bones) if rig is not None else 0
                except Exception:
                    rig_ct = -1

                try:
                    tri_ct = len(mesh.triangles) if mesh is not None else 0
                    v_ct = len(mesh.vertices) if mesh is not None else 0
                except Exception:
                    tri_ct, v_ct = -1, -1

                self_inner._dbg(
                    f"[viewer] redraw w={w} h={h} pose_ct={pose_ct} rig_ct={rig_ct} mesh={'yes' if mesh else 'no'} "
                    f"verts={v_ct} tris={tri_ct} cam_center={getattr(self_inner,'_cam_center',None)} dist={getattr(self_inner,'_cam_dist',None)}"
                )

                if pose is None or rig is None:
                    self_inner._dbg("[viewer] redraw: pose or rig is None -> nothing to draw")
                    glFlush()
                    return

                # One-shot auto-center (cheap)
                if getattr(self_inner, "_auto_center", False):
                    xs = [p[0] for p in pose.world_pos.values()]
                    ys = [p[1] for p in pose.world_pos.values()]
                    zs = [p[2] for p in pose.world_pos.values()]
                    if xs and ys and zs:
                        self_inner._cam_center = (
                            (min(xs) + max(xs)) * 0.5,
                            (min(ys) + max(ys)) * 0.5,
                            (min(zs) + max(zs)) * 0.5,
                        )
                    self_inner._auto_center = False

                # Camera transform:
                # translate back by dist, rotate by inverse orientation, translate by -center
                cx, cy, cz = self_inner._cam_center
                dist = float(self_inner._cam_dist)

                glTranslatef(0.0, 0.0, -dist)

                axis, angle_deg = GLViewerFrame._quat_to_axis_angle(self_inner._cam_rot_q)
                if angle_deg != 0.0:
                    # inverse for view transform
                    glRotatef(-float(angle_deg), float(axis[0]), float(axis[1]), float(axis[2]))

                glTranslatef(-float(cx), -float(cy), -float(cz))
                self_inner._gl_check("after camera transform")



                mesh = getattr(self_inner, "_mesh", None)

                # Support wrapper meshes that contain multiple geosets
                meshes_to_draw = []
                if mesh is not None and getattr(mesh, "submeshes", None):
                    meshes_to_draw = list(getattr(mesh, "submeshes") or [])
                elif mesh is not None:
                    meshes_to_draw = [mesh]

                enabled = getattr(self_inner, "_enabled_geosets", None)
                if enabled and len(enabled) == len(meshes_to_draw):
                    meshes_to_draw = [m for m, ok in zip(meshes_to_draw, enabled) if ok]

                for mesh in meshes_to_draw:
                    if mesh is None:
                        continue
                    try:
                        from .evaluator import transform_point

                        verts = mesh.vertices
                        tris = mesh.triangles
                        uvs = getattr(mesh, "uvs", None)
                        tex_name = getattr(mesh, "texture_name", None)

                        vgroups = mesh.vertex_groups
                        gmat = mesh.groups_matrices

                        def bones_for_vertex(vid: int) -> Optional[list[int]]:
                            if vgroups is None or gmat is None:
                                return None
                            if vid < 0 or vid >= len(vgroups):
                                return None

                            gi = vgroups[vid]
                            if gi < 0 or gi >= len(gmat):
                                return None

                            mats = gmat[gi]
                            if not mats:
                                return None

                            return [int(x) for x in mats]

                        def skin_vertex(v: tuple[float, float, float], vid: int) -> tuple[float, float, float]:
                            bids = bones_for_vertex(vid)
                            if not bids:
                                return v

                            accx = accy = accz = 0.0
                            n = 0
                            for bid in bids:
                                mtx = pose.world_mats.get(bid)
                                if mtx is None:
                                    continue
                                tv = transform_point(mtx, v)
                                accx += float(tv[0])
                                accy += float(tv[1])
                                accz += float(tv[2])
                                n += 1
                            if n <= 0:
                                return v
                            inv = 1.0 / float(n)
                            return (accx * inv, accy * inv, accz * inv)

                        tex_id = None
                        if tex_name and uvs:
                            tex_id = self_inner._get_texture_id(tex_name)

                        if tex_id is not None:
                            glEnable(GL_TEXTURE_2D)
                            glBindTexture(GL_TEXTURE_2D, tex_id)
                        else:
                            glDisable(GL_TEXTURE_2D)

                        uvs = getattr(mesh, "uvs", None)
                        materials = getattr(mesh, "materials", None)
                        textures = getattr(mesh, "textures", None)
                        mid = getattr(mesh, "geoset_material_id", None)

                        layers = None
                        if (
                            uvs is not None
                            and materials is not None
                            and textures is not None
                            and mid is not None
                            and 0 <= int(mid) < len(materials)
                        ):
                            layers = materials[int(mid)].get("layers") or None

                        # fallback: if we don't have layers, keep old behavior (solid color)
                        if not layers:
                            glColor3f(0.35, 0.7, 0.35)
                            glBegin(GL_TRIANGLES)
                            for (i0, i1, i2) in tris:
                                for vid in (i0, i1, i2):
                                    v = verts[vid]
                                    v = skin_vertex(v, vid)
                                    glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                            glEnd()
                        else:
                            # Depth test ON for real mesh rendering
                            glEnable(GL_DEPTH_TEST)

                            for li, layer in enumerate(layers):
                                tex_id = layer.get("texture_id", None)
                                filter_mode = layer.get("filter_mode", "None")
                                alpha = float(layer.get("alpha", 1.0) or 1.0)

                                # ---- THIS IS THE IMPORTANT WIRING ----
                                self_inner._apply_filter_mode(filter_mode, alpha)

                                # resolve/bind texture for this layer (you implement this)
                                glDisable(GL_TEXTURE_2D)
                                tid = None
                                if tex_id is not None and 0 <= int(tex_id) < len(textures):
                                    tid = self_inner._get_texture_id_from_texture_entry(textures[int(tex_id)])
                                if tid is not None:
                                    glEnable(GL_TEXTURE_2D)
                                    glBindTexture(GL_TEXTURE_2D, tid)

                                # When textured, use white so the texture shows un-tinted
                                glColor4f(1.0, 1.0, 1.0, float(alpha))

                                glBegin(GL_TRIANGLES)
                                for (i0, i1, i2) in tris:
                                    for vid in (i0, i1, i2):
                                        v = verts[vid]
                                        v = skin_vertex(v, vid)

                                        # UVs (flip V if needed later)
                                        if uvs is not None and vid < len(uvs):
                                            u, vv = uvs[vid]
                                            glTexCoord2f(float(u), float(vv))

                                        glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                                glEnd()

                            # restore defaults after mesh
                            glDisable(GL_TEXTURE_2D)
                            glDisable(GL_BLEND)
                            glDepthMask(True)
                            glDisable(GL_DEPTH_TEST)

                        if tex_id is not None:
                            glDisable(GL_TEXTURE_2D)

                    except Exception as e:
                        print(f"[viewer] EXCEPTION during mesh draw: {e!r}")
                self_inner._gl_check("after mesh draw")
                
                # --- BONES ---
                if getattr(self_inner, "_show_bones", True):
                    active_ids = getattr(self_inner, "_active_ids", None)
                    edge_count = 0

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
                        edge_count += 1

                        if active_ids is not None and oid in active_ids:
                            glColor3f(1.0, 0.8, 0.2)
                        else:
                            glColor3f(0.7, 0.7, 0.9)

                        glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                        glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                    glEnd()

                    if edge_count == 0 and not getattr(self_inner, "_warned_zero_edges", False):
                        print("WARNING: drew 0 bone edges (check parent_id wiring)")
                        self_inner._warned_zero_edges = True

                self_inner._gl_check("after bone draw")
                glFlush()

        self._impl = _Impl(self, width=640, height=480)
        self._impl.pack(fill="both", expand=True)

        # IMPORTANT: event-driven redraw for performance
        self._impl.animate = 0

        # initial values
        self._impl._pose = None
        self._impl._rig = None
        self._impl._active_ids = None

        # mouse bindings (orbit/pan/zoom)
        self._impl.bind("<ButtonPress-3>", self._impl._begin_orbit)
        self._impl.bind("<B3-Motion>", self._impl._on_drag)
        self._impl.bind("<ButtonRelease-3>", self._impl._end_drag)

        # Shift + MMB pan (Tk uses Button-2 for middle)
        self._impl.bind("<Shift-ButtonPress-2>", self._impl._begin_pan)
        self._impl.bind("<Shift-B2-Motion>", self._impl._on_drag)
        self._impl.bind("<ButtonRelease-2>", self._impl._end_drag)

        # wheel (Windows / macOS)
        self._impl.bind("<MouseWheel>", self._impl._on_wheel)
        # wheel (many Linux setups)
        self._impl.bind("<Button-4>", lambda _e: self._impl._on_wheel(type("E", (), {"delta": 120})()))
        self._impl.bind("<Button-5>", lambda _e: self._impl._on_wheel(type("E", (), {"delta": -120})()))

        # reset (Ctrl+R) - optional; ViewerWindow also binds this
        self._impl.bind("<Control-r>", self._impl._on_reset_key)
        self._impl.bind("<Control-R>", self._impl._on_reset_key)

    # ---- public camera helpers (ViewerWindow can call these) ----
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

    def set_pose(self, pose: Pose, rig: Rig, active_ids: Optional[set[int]] = None) -> None:
        if self._impl is None:
            return
        self._impl._pose = pose
        self._impl._rig = rig
        self._impl._active_ids = active_ids
        self._impl.request_redraw()

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