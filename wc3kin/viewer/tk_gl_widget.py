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

try:
    from OpenGL.GL import (
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,      # kept imported (not used by default path)
        glFrustum,
        glVertex3f,
        glViewport,
        glEnable,
        glHint,
        glDisable,
        glFlush,
        glTranslatef,
        glRotatef,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_LINES,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_DEPTH_TEST,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_NICEST,
    )
except Exception as e:  # pragma: no cover
    # If PyOpenGL isn't installed, we will surface a clear error on init.
    _PYOPENGL_IMPORT_ERR = e
else:
    _PYOPENGL_IMPORT_ERR = None


class GLViewerFrame(tk.Frame):
    """
    Wrapper frame that either hosts the real OpenGL widget (OpenGLFrame),
    or shows a helpful error message if deps are missing.

    Adds a simple orbit camera:
      - RMB drag: orbit (inverted X/Y)
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
            _cam_yaw_deg: float
            _cam_pitch_deg: float
            _cam_dist: float
            _cam_default: Dict[str, Any]
            _auto_center: bool

            # drag bookkeeping
            _drag_mode: str  # "orbit" | "pan" | ""
            _drag_last_xy: Tuple[int, int]

            def initgl(self_inner) -> None:
                glClearColor(0.08, 0.08, 0.10, 1.0)
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glDisable(GL_DEPTH_TEST)  # skeleton lines only; can be enabled later

                # camera defaults
                self_inner._cam_center = (0.0, 0.0, 0.0)
                self_inner._cam_yaw_deg = 35.0
                self_inner._cam_pitch_deg = -20.0
                self_inner._cam_dist = 900.0
                self_inner._cam_default = {}
                self_inner._auto_center = True

                self_inner._drag_mode = ""
                self_inner._drag_last_xy = (0, 0)

            def request_redraw(self_inner) -> None:
                """
                Ask OpenGLFrame to draw a frame using whatever internal method
                ensures the GL context is current.
                """
                if hasattr(self_inner, "_display"):
                    self_inner.after_idle(self_inner._display)  # type: ignore[attr-defined]
                elif hasattr(self_inner, "tkRedraw"):
                    self_inner.after_idle(self_inner.tkRedraw)  # type: ignore[attr-defined]
                else:
                    self_inner.after_idle(self_inner.redraw)

            # -------- camera API --------
            def get_camera_state(self_inner) -> Dict[str, Any]:
                cx, cy, cz = self_inner._cam_center
                return {
                    "center": [float(cx), float(cy), float(cz)],
                    "yaw_deg": float(self_inner._cam_yaw_deg),
                    "pitch_deg": float(self_inner._cam_pitch_deg),
                    "dist": float(self_inner._cam_dist),
                }

            def set_camera_state(self_inner, state: Dict[str, Any]) -> None:
                c = state.get("center") or [0.0, 0.0, 0.0]
                if isinstance(c, (list, tuple)) and len(c) >= 3:
                    self_inner._cam_center = (float(c[0]), float(c[1]), float(c[2]))
                if "yaw_deg" in state:
                    self_inner._cam_yaw_deg = float(state["yaw_deg"])
                if "pitch_deg" in state:
                    self_inner._cam_pitch_deg = float(state["pitch_deg"])
                if "dist" in state:
                    self_inner._cam_dist = max(5.0, float(state["dist"]))
                self_inner._auto_center = False

            def snapshot_default_camera(self_inner) -> None:
                if not self_inner._cam_default:
                    self_inner._cam_default = self_inner.get_camera_state()

            def reset_camera(self_inner) -> None:
                if self_inner._cam_default:
                    # note: set_camera_state disables auto-center; we want that for reset
                    self_inner.set_camera_state(dict(self_inner._cam_default))
                else:
                    self_inner._cam_center = (0.0, 0.0, 0.0)
                    self_inner._cam_yaw_deg = 35.0
                    self_inner._cam_pitch_deg = -20.0
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
                # heuristic for ~45deg fov: keep model in view
                self_inner._cam_dist = max(50.0, float(span) * 1.8)
                self_inner._auto_center = True
                self_inner.request_redraw()

            # -------- input handling --------
            def _begin_orbit(self_inner, e: tk.Event) -> None:
                self_inner._drag_mode = "orbit"
                self_inner._drag_last_xy = (int(e.x), int(e.y))
                self_inner._auto_center = False

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
                    # inverted X/Y requested
                    self_inner._cam_yaw_deg -= float(dx) * 0.35
                    self_inner._cam_pitch_deg -= float(dy) * 0.35
                    self_inner._cam_pitch_deg = GLViewerFrame._clamp(self_inner._cam_pitch_deg, -89.0, 89.0)
                    self_inner.request_redraw()
                    return

                if mode == "pan":
                    # pan in camera screen plane
                    dist = float(self_inner._cam_dist)
                    pan_scale = dist * 0.0025

                    yaw = math.radians(self_inner._cam_yaw_deg)
                    pit = math.radians(self_inner._cam_pitch_deg)

                    fx = math.cos(pit) * math.sin(yaw)
                    fy = math.sin(pit)
                    fz = math.cos(pit) * math.cos(yaw)
                    fwd = GLViewerFrame._normalize((fx, fy, fz))
                    up = (0.0, 1.0, 0.0)
                    right = GLViewerFrame._normalize(GLViewerFrame._cross(fwd, up))
                    cam_up = GLViewerFrame._normalize(GLViewerFrame._cross(right, fwd))

                    cx, cy, cz = self_inner._cam_center
                    # inverted screen mapping
                    cx -= right[0] * dx * pan_scale
                    cy -= right[1] * dx * pan_scale
                    cz -= right[2] * dx * pan_scale
                    cx += cam_up[0] * dy * pan_scale
                    cy += cam_up[1] * dy * pan_scale
                    cz += cam_up[2] * dy * pan_scale
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

                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect = float(w) / float(h)

                # simple perspective frustum (avoid GLU dependency)
                fov_deg = 45.0
                z_near = 5.0
                z_far = 200000.0
                top = math.tan(math.radians(fov_deg * 0.5)) * z_near
                right = top * aspect
                glFrustum(-right, right, -top, top, z_near, z_far)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()

                # camera orbit around center
                cx, cy, cz = self_inner._cam_center
                yaw = math.radians(self_inner._cam_yaw_deg)
                pit = math.radians(self_inner._cam_pitch_deg)
                dist = float(self_inner._cam_dist)

                cam_x = cx + dist * (math.cos(pit) * math.sin(yaw))
                cam_y = cy + dist * (math.sin(pit))
                cam_z = cz + dist * (math.cos(pit) * math.cos(yaw))

                # view transform
                glRotatef(-self_inner._cam_pitch_deg, 1.0, 0.0, 0.0)
                glRotatef(-self_inner._cam_yaw_deg, 0.0, 1.0, 0.0)
                glTranslatef(-cam_x, -cam_y, -cam_z)

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                pose = getattr(self_inner, "_pose", None)
                rig = getattr(self_inner, "_rig", None)
                if pose is None or rig is None:
                    glFlush()
                    return

                # allow center to follow animation until user moves camera
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

                # --- BONES ---
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

                glFlush()

        self._impl = _Impl(self, width=640, height=480)
        self._impl.pack(fill="both", expand=True)

        # let pyopengltk drive redraws properly
        self._impl.animate = 1

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

        # reset (Ctrl+R)
        self._impl.bind("<Control-r>", self._impl._on_reset_key)
        self._impl.bind("<Control-R>", self._impl._on_reset_key)

        # allow key events to reach the widget
        try:
            self._impl.focus_set()
        except Exception:
            pass

    # ---- public camera helpers (ViewerWindow can call these) ----
    def get_camera_state(self) -> Optional[Dict[str, Any]]:
        if self._impl is None:
            return None
        return self._impl.get_camera_state()

    def set_camera_state(self, state: Dict[str, Any]) -> None:
        if self._impl is None:
            return
        self._impl.set_camera_state(state)

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
        # no redraw call needed; animate loop will call redraw()
