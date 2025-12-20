from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import sqlite3
from pathlib import Path

from .evaluator import UnitAnimEvaluator, build_anims_from_boneanims_json, build_rig_from_bones_json
from .tk_gl_widget import GLViewerFrame
from .types import SequenceDef
from .. import db as dbmod


class ViewerWindow(tk.Toplevel):
    """
    Minimal viewer:
      - skeleton render
      - play/pause/rewind
      - loop toggle
    """

    TICK_MS = 16  # ~60fps stepping; deterministic step size

    def __init__(
        self,
        master: tk.Misc,
        *,
        con: sqlite3.Connection,
        units_root: Path,
        unit_id: int,
        sequence_name: str,
    ) -> None:
        super().__init__(master)
        self.title("WC3 Viewer")
        self.geometry("980x680")

        self.con = con
        self.units_root = units_root
        self.unit_id = int(unit_id)
        self.sequence_name = str(sequence_name)

        self.playing = False
        self.loop_var = tk.BooleanVar(value=True)

        self.seq: Optional[SequenceDef] = None
        self.t_ms: int = 0

        self._evaluator: Optional[UnitAnimEvaluator] = None
        self._rig = None

        # layout
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)

        self.gl = GLViewerFrame(top)
        self.gl.pack(fill="both", expand=True, padx=8, pady=8)

        controls = ttk.Frame(top)
        controls.pack(fill="x", padx=8, pady=(0, 8))

        self.play_btn = ttk.Button(controls, text="Play", command=self._on_play)
        self.pause_btn = ttk.Button(controls, text="Pause", command=self._on_pause)
        self.rewind_btn = ttk.Button(controls, text="Rewind", command=self._on_rewind)

        self.play_btn.pack(side="left")
        self.pause_btn.pack(side="left", padx=(6, 0))
        self.rewind_btn.pack(side="left", padx=(6, 0))

        ttk.Checkbutton(controls, text="Loop", variable=self.loop_var).pack(side="left", padx=(12, 0))

        self.time_lbl = ttk.Label(controls, text="t=0ms")
        self.time_lbl.pack(side="right")

        # load and render first frame immediately
        self._load_from_db()
        self._render_current()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.playing = False
        self.destroy()

    def _on_play(self) -> None:
        if self._evaluator is None or self.seq is None:
            return
        if not self.playing:
            self.playing = True
            self._tick()

    def _on_pause(self) -> None:
        self.playing = False

    def _on_rewind(self) -> None:
        if self.seq is None:
            return
        self.t_ms = int(self.seq.start_ms)
        self._render_current()

    def _tick(self) -> None:
        if not self.playing or self.seq is None or self._evaluator is None:
            return

        self.t_ms += self.TICK_MS
        if self.t_ms > self.seq.end_ms:
            if self.loop_var.get():
                self.t_ms = int(self.seq.start_ms)
            else:
                self.t_ms = int(self.seq.end_ms)
                self.playing = False

        self._render_current()
        self.after(self.TICK_MS, self._tick)

    def _load_from_db(self) -> None:
        # Ensure sequence exists
        seq = dbmod.get_sequence_detail(self.con, self.unit_id, self.sequence_name)
        if seq is None:
            raise RuntimeError(f"Sequence not found in DB for unit_id={self.unit_id}: {self.sequence_name}")
        self.seq = seq
        self.t_ms = int(seq.start_ms)

        # Need harvested blobs
        bones_json = dbmod.get_harvested_json_blob(self.con, self.unit_id, "bones")
        boneanims_json = dbmod.get_harvested_json_blob(self.con, self.unit_id, "boneanims")
        if bones_json is None or boneanims_json is None:
            raise RuntimeError(
                "Missing harvested JSON blobs in DB.\n"
                "Expected kinds: 'bones' and 'boneanims'.\n"
                "Use blob ingest (best-effort) before opening viewer."
            )

        rig = build_rig_from_bones_json(bones_json)
        anims = build_anims_from_boneanims_json(boneanims_json)
        self._rig = rig
        self._evaluator = UnitAnimEvaluator(rig=rig, anims=anims)

    def _render_current(self) -> None:
        if self._evaluator is None or self._rig is None:
            return
        pose = self._evaluator.evaluate_pose(self.t_ms)
        self.gl.set_pose(pose, self._rig, active_ids=None)
        if not hasattr(self, "_fit_done"):
            xs = [p[0] for p in pose.world_pos.values()]
            ys = [p[1] for p in pose.world_pos.values()]
            zs = [p[2] for p in pose.world_pos.values()]

            if xs:
                span = max(
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    max(zs) - min(zs),
                )
                self.gl._impl._ortho_scale = max(span * 0.75, 100.0)

            self._fit_done = True
        self.time_lbl.config(text=f"t={self.t_ms}ms")
