from __future__ import annotations

import math
import tkinter as tk
from typing import Optional

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
        glOrtho,
        glVertex3f,
        glViewport,
        glEnable,
        glHint,
        glDisable,
        glFlush,
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
    """

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
            def initgl(self_inner) -> None:
                glClearColor(0.08, 0.08, 0.10, 1.0)
                glEnable(GL_LINE_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glDisable(GL_DEPTH_TEST)  # skeleton lines only; depth can be added later

            def request_redraw(self_inner) -> None:
                """
                Ask OpenGLFrame to draw a frame using whatever internal method
                ensures the GL context is current.
                """
                # Different pyopengltk versions expose different internals.
                if hasattr(self_inner, "_display"):
                    self_inner.after_idle(self_inner._display)  # type: ignore[attr-defined]
                elif hasattr(self_inner, "tkRedraw"):
                    self_inner.after_idle(self_inner.tkRedraw)  # type: ignore[attr-defined]
                else:
                    # Worst-case fallback (may not have context): schedule via update_idletasks first
                    self_inner.after_idle(self_inner.redraw)

            def redraw(self_inner) -> None:
                if getattr(self_inner, "_pose", None) is None or getattr(self_inner, "_rig", None) is None:
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glFlush()
                    return

                w = int(self_inner.winfo_width())
                h = int(self_inner.winfo_height())
                if w <= 1 or h <= 1:
                    return
                glViewport(0, 0, w, h)

                # Basic fixed camera:
                # orthographic projection with auto-scale-ish; later you'll add real camera controls.
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect = float(w) / float(h)
                # units are in wc3 coordinates; start with a generous box
                s = getattr(self_inner, "_ortho_scale", 650.0)
                glOrtho(-s * aspect, s * aspect, -s, s, -2000.0, 2000.0)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                pose: Pose = self_inner._pose
                rig: Rig = self_inner._rig
                active_ids = getattr(self_inner, "_active_ids", None)

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

                    if active_ids is not None and oid in active_ids:
                        glColor3f(1.0, 0.8, 0.2)  # highlight
                    else:
                        glColor3f(0.7, 0.7, 0.9)

                    glVertex3f(float(p0[0]), float(p0[1]), float(p0[2]))
                    glVertex3f(float(p1[0]), float(p1[1]), float(p1[2]))
                glEnd()
                glFlush()

        self._impl = _Impl(self, width=640, height=480)
        self._impl.pack(fill="both", expand=True)

        try:
            self._impl.request_redraw()
        except Exception:
            pass

        # initial projection scale
        self._impl._ortho_scale = 650.0
        self._impl._pose = None
        self._impl._rig = None
        self._impl._active_ids = None
    
    def set_pose(self, pose: Pose, rig: Rig, active_ids: Optional[set[int]] = None) -> None:
        if self._impl is None:
            return
        self._impl._pose = pose
        self._impl._rig = rig
        self._impl._active_ids = active_ids
        self._impl.request_redraw()
