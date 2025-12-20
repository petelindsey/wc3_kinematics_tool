from __future__ import annotations

import tkinter as tk

from pyopengltk import OpenGLFrame
from OpenGL.GL import (
    glBegin, glClear, glClearColor, glColor3f, glEnd, glFlush, glLineWidth,
    glLoadIdentity, glMatrixMode, glOrtho, glVertex2f, glViewport,
    glGetString, GL_VERSION, GL_RENDERER, GL_VENDOR,
    GL_COLOR_BUFFER_BIT, GL_LINES, GL_MODELVIEW, GL_PROJECTION,
)

class TestFrame(OpenGLFrame):
    def initgl(self) -> None:
        # If you see these prints, the GL context is actually created.
        try:
            print("GL_VENDOR  :", glGetString(GL_VENDOR))
            print("GL_RENDERER:", glGetString(GL_RENDERER))
            print("GL_VERSION :", glGetString(GL_VERSION))
        except Exception as e:
            print("glGetString failed:", repr(e))

        glClearColor(0.05, 0.05, 0.08, 1.0)

    def redraw(self) -> None:
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        glViewport(0, 0, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT)

        glLineWidth(6.0)
        glBegin(GL_LINES)

        # red horizontal
        glColor3f(1.0, 0.0, 0.0)
        glVertex2f(-0.9, 0.0)
        glVertex2f(0.9, 0.0)

        # green vertical
        glColor3f(0.0, 1.0, 0.0)
        glVertex2f(0.0, -0.9)
        glVertex2f(0.0, 0.9)

        glEnd()
        glFlush()

def main() -> None:
    root = tk.Tk()
    root.title("pyopengltk smoke test (animate loop)")
    root.geometry("800x600")

    frame = TestFrame(root, width=800, height=600)
    frame.pack(fill="both", expand=True)

    # This is the key: let pyopengltk drive redraws.
    frame.animate = 1

    root.mainloop()

if __name__ == "__main__":
    main()
