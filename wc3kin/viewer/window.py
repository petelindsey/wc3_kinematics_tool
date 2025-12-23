from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import sqlite3
from pathlib import Path

from .evaluator import UnitAnimEvaluator, build_anims_from_boneanims_json
from .tk_gl_widget import GLViewerFrame
from .types import SequenceDef
from .. import db as dbmod

from wc3kin.viewer.mdl_nodes import load_nodes_from_mdl
from wc3kin.viewer.evaluator import build_rig_from_mdl_nodes

from .view_persistence import ViewerPersist, default_persistence_path
from .mesh_provider import MdlFileMeshProvider

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
        # loop off by default
        self.loop_var = tk.BooleanVar(value=False)

        self.seq: Optional[SequenceDef] = None
        self.t_ms: int = 0

        self._evaluator: Optional[UnitAnimEvaluator] = None
        self._rig = None

        # persistence (camera)
        self._persist = ViewerPersist.load(default_persistence_path(self.con))
        self._cam_init_done = False

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

        # Ctrl+R camera reset (bind to this window; avoid bind_all)
        self.bind("<Control-r>", lambda _e: self._reset_camera())
        self.bind("<Control-R>", lambda _e: self._reset_camera())

        # load and render first frame immediately
        self._load_from_db()
        self._render_current()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _reset_camera(self) -> None:
        try:
            if self.gl is not None:
                self.gl.reset_camera()
        except Exception:
            pass

    def _on_close(self) -> None:
        self.playing = False

        # save camera state
        try:
            cam = self.gl.get_camera_state() if self.gl is not None else None
            if cam:
                self._persist.set_camera(self.unit_id, self.sequence_name, cam)
                self._persist.save()
        except Exception:
            pass

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
                # stop + auto-rewind
                self.t_ms = int(self.seq.start_ms)
                self.playing = False

        self._render_current()
        self.after(self.TICK_MS, self._tick)

    def _load_from_db(self) -> None:
        print("load from db called")
        self._mesh = None
        self._mesh_provider = None
        seq = dbmod.get_sequence_detail(self.con, self.unit_id, self.sequence_name)
        if seq is None:
            raise RuntimeError(f"Sequence not found in DB for unit_id={self.unit_id}: {self.sequence_name}")
        self.seq = seq
        self.t_ms = int(seq.start_ms)

        bones_json = dbmod.get_harvested_json_blob(self.con, self.unit_id, "bones")
        boneanims_json = dbmod.get_harvested_json_blob(self.con, self.unit_id, "boneanims")
        print("processing bones json")
        if bones_json is None or boneanims_json is None:
            raise RuntimeError(
                "Missing harvested JSON blobs in DB.\n"
                "Expected kinds: 'bones' and 'boneanims'.\n"
                "Use blob ingest (best-effort) before opening viewer."
            )

        mdl_path_str = bones_json.get("mdl")
        if not mdl_path_str:
            print("bones_json missing 'mdl' path; cannot load MDL for viewer rig.")
            raise RuntimeError("bones_json missing 'mdl' path; cannot load MDL for viewer rig.")
        mdl_path = Path(mdl_path_str)

        if not mdl_path.exists():
            print(f"MDL path does not exist on disk: {mdl_path}")
            raise RuntimeError(f"MDL path does not exist on disk: {mdl_path}")
        
        try:
            from .mesh_provider import MdlFileMeshProvider

            print(f"[viewer] MDL path from bones_json['mdl'] = {mdl_path}")
            print(f"[viewer] MDL exists={mdl_path.exists()} size={mdl_path.stat().st_size}")

            self._mesh_provider = MdlFileMeshProvider(mdl_path=mdl_path)
            self._mesh = self._mesh_provider.load_mesh(con=self.con, unit_id=self.unit_id)

            if self._mesh is None:
                print("[viewer] Mesh provider returned None (bones-only)")
            else:
                self.gl.set_mesh(self._mesh)

                m = self._mesh
                print(
                    "[viewer] Mesh loaded:"
                    f" verts={len(m.vertices)} tris={len(m.triangles)}"
                    f" vgroups={'yes' if m.vertex_groups else 'no'}"
                    f" groups_matrices={'yes' if m.groups_matrices else 'no'}"
                )
        except Exception as e:
            self._mesh = None
            print(f"[viewer] Mesh load failed (bones-only): {e!r}")

        print(f"[viewer] MDL path from bones_json['mdl'] = {mdl_path}")
        print(f"[viewer] MDL exists={mdl_path.exists()} size={mdl_path.stat().st_size if mdl_path.exists() else 'n/a'}")
        m = self._mesh
        if m is not None:
            print(
                "[viewer] Mesh loaded:"
                f" verts={len(m.vertices)} tris={len(m.triangles)}"
                f" vgroups={'yes' if m.vertex_groups else 'no'}"
                f" groups_matrices={'yes' if m.groups_matrices else 'no'}"
            )
            if m.vertex_groups:
                print(f"[viewer] vertex_groups count={len(m.vertex_groups)} min={min(m.vertex_groups)} max={max(m.vertex_groups)}")
            if m.groups_matrices:
                nonempty = sum(1 for g in m.groups_matrices if g)
                maxlen = max((len(g) for g in m.groups_matrices), default=0)
                print(f"[viewer] groups_matrices count={len(m.groups_matrices)} nonempty={nonempty} max_group_len={maxlen}")

        self._mesh_provider = MdlFileMeshProvider(mdl_path=mdl_path)
        self._mesh = self._mesh_provider.load_mesh(con=self.con, unit_id=self.unit_id)

        # pass mesh into GL widget so redraw can render it
        self.gl.set_mesh(self._mesh)


        print("Building Nodes and Rig")
        nodes_by_id = load_nodes_from_mdl(mdl_path)
        rig = build_rig_from_mdl_nodes(nodes_by_id)

        anims = build_anims_from_boneanims_json(boneanims_json)

        # --- DEBUG: do boneanim key times line up with sequence times? ---
        self._time_offset = 0
        print("Building All Times")
        all_times = []
        for ch in anims.values():
            all_times.extend([k.time_ms for k in ch.translation])
            all_times.extend([k.time_ms for k in ch.rotation])
            all_times.extend([k.time_ms for k in ch.scaling])
        print("Checking if All Times is empty")
        try:
            if all_times:
                keys_min, keys_max = min(all_times), max(all_times)
                seq_dur = int(self.seq.end_ms - self.seq.start_ms)

                if keys_min >= 0 and keys_max <= seq_dur + 2 and int(self.seq.start_ms) != 0:
                    self._time_offset = int(self.seq.start_ms)
                    print(f"[viewer] using relative key times; offset={self._time_offset}ms")
            print(
                f"[viewer] seq={self.seq.name} start={self.seq.start_ms} end={self.seq.end_ms} "
                f"dur={int(self.seq.end_ms - self.seq.start_ms)}"
            )
            print(
                f"[viewer] key_times: empty={not bool(all_times)}"
                + (f" min={min(all_times)} max={max(all_times)} count={len(all_times)}" if all_times else "")
            )
        except Exception as e:
            print(f"Load From DB Error:{e}")

        self._rig = rig
        self._evaluator = UnitAnimEvaluator(rig=rig, anims=anims)

    def _render_current(self) -> None:
        if self._evaluator is None or self._rig is None:
            return

        offset = int(getattr(self, "_time_offset", 0))
        t_sample = self.t_ms - offset

        # NOTE: if you intended to use offset sampling, switch to evaluate_pose(t_sample)
        pose = self._evaluator.evaluate_pose(self.t_ms)

        # First render: restore persisted camera or do a default fit
        if not self._cam_init_done:
            restored = None
            try:
                restored = self._persist.get_camera(self.unit_id, self.sequence_name)
            except Exception:
                restored = None

            if restored:
                self.gl.set_camera_state(restored)
            else:
                self.gl.fit_camera_to_pose(pose)

            # snapshot default camera (for Ctrl+R)
            self.gl.snapshot_default_camera()
            self._cam_init_done = True

        self.gl.set_pose(pose, self._rig, active_ids=None)
        self.time_lbl.config(text=f"t={self.t_ms}ms")
