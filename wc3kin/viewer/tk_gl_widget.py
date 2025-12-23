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

from .mesh_provider import MeshData

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

                pose = getattr(self_inner, "_pose", None)
                rig = getattr(self_inner, "_rig", None)
                if pose is None or rig is None:
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

                mesh = getattr(self_inner, "_mesh", None)
                if mesh is not None:
                    from .evaluator import transform_point

                    verts = mesh.vertices
                    tris = mesh.triangles
                    vgroups = mesh.vertex_groups
                    gmat = mesh.groups_matrices

                    def bone_for_vertex(vid: int) -> Optional[int]:
                        if not vgroups or not gmat:
                            return None
                        if vid < 0 or vid >= len(vgroups):
                            return None
                        gi = vgroups[vid]
                        if gi < 0 or gi >= len(gmat):
                            return None
                        mats = gmat[gi]
                        if not mats:
                            return None
                        return int(mats[0])

                    glColor3f(0.35, 0.7, 0.35)
                    glBegin(GL_TRIANGLES)
                    for (i0, i1, i2) in tris:
                        for vid in (i0, i1, i2):
                            v = verts[vid]
                            bid = bone_for_vertex(vid)
                            if bid is not None and bid in pose.world_mats:
                                v = transform_point(pose.world_mats[bid], v)
                            glVertex3f(float(v[0]), float(v[1]), float(v[2]))
                    glEnd()

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
    
    def set_show_bones(self, show: bool) -> None:
        if self._impl is None:
            return
        self._impl.set_show_bones(show)

    def get_show_bones(self) -> bool:
        if self._impl is None:
            return True
        return self._impl.get_show_bones()
