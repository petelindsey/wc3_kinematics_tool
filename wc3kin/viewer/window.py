#wc3kin/viewer/window.py
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
from .evaluator import build_anims_from_mdl

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
        # ---- Debug UI vars ----
        self.dbg_alpha_off_var = tk.BooleanVar(value=False)
        self.dbg_disable_textures_var = tk.BooleanVar(value=False)
        self.dbg_color_by_tri_var = tk.BooleanVar(value=False)
        self.dbg_prints_var = tk.BooleanVar(value=True)
        self.dbg_flip_v_var = tk.BooleanVar(value=True)
        self.teamcolor_mode_var = tk.StringVar(value="wc3_mask")
        self.teamcolor_blend_var = tk.StringVar(value="layer")

        #view bones off by default
        self.bones_var = tk.BooleanVar(value=False)

        #number of player for team color
        self.player_var = tk.IntVar(value=0)
        self._geoset_vars = []  # list[tk.BooleanVar]
        self._geosets_popup = None

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
        self._on_debug_flags_changed()
        try:
            self.gl.set_show_bones(bool(self.bones_var.get()))
        except Exception:
            pass

        try:
            self.gl.set_player_index(int(self.player_var.get()))
        except Exception:
            pass

        controls = ttk.Frame(top)
        controls.pack(fill="x", padx=8, pady=(0, 8))

        menubar = tk.Menu(self)
        self.config(menu=menubar)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)

        debug_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Debug", menu=debug_menu)

        # Debug toggles
        debug_menu.add_checkbutton(
            label="Alpha Off (Force Opaque)",
            variable=self.dbg_alpha_off_var,
            command=self._on_debug_flags_changed,
        )
        debug_menu.add_checkbutton(
            label="Disable Textures",
            variable=self.dbg_disable_textures_var,
            command=self._on_debug_flags_changed,
        )
        debug_menu.add_checkbutton(
            label="Color By Triangle",
            variable=self.dbg_color_by_tri_var,
            command=self._on_debug_flags_changed,
        )
        debug_menu.add_checkbutton(
            label="Flip V (UV)",
            variable=self.dbg_flip_v_var,
            command=self._on_debug_flags_changed,
        )
        debug_menu.add_separator()
        debug_menu.add_checkbutton(
            label="Debug Prints",
            variable=self.dbg_prints_var,
            command=self._on_debug_flags_changed,
        )
        debug_menu.add_separator()

        team_menu = tk.Menu(debug_menu, tearoff=0)
        debug_menu.add_cascade(label="TeamColor", menu=team_menu)

        mode_menu = tk.Menu(team_menu, tearoff=0)
        team_menu.add_cascade(label="Mode", menu=mode_menu)

        for label, val in [
            ("WC3 Mask (RGB=team, A=texA*alpha)", "wc3_mask"),
            ("Modulate (RGB=tex*team)", "modulate"),
            ("Replace RGB, Keep Alpha", "replace_rgb_keep_alpha"),
            ("Off (treat as normal)", "off"),
        ]:
            mode_menu.add_radiobutton(
                label=label,
                variable=self.teamcolor_mode_var,
                value=val,
                command=self._on_debug_flags_changed,
            )

        blend_menu = tk.Menu(team_menu, tearoff=0)
        team_menu.add_cascade(label="Blend", menu=blend_menu)

        for label, val in [
            ("Layer (material filter mode)", "layer"),
            ("Force Alpha (SRC_A, 1-SRC_A)", "alpha"),
            ("Force Add (SRC_A, ONE)", "add"),
            ("None (no blend/test)", "none"),
        ]:
            blend_menu.add_radiobutton(
                label=label,
                variable=self.teamcolor_blend_var,
                value=val,
                command=self._on_debug_flags_changed,
            )
        self.play_btn = ttk.Button(controls, text="Play", command=self._on_play)
        self.pause_btn = ttk.Button(controls, text="Pause", command=self._on_pause)
        self.rewind_btn = ttk.Button(controls, text="Rewind", command=self._on_rewind)

        self.play_btn.pack(side="left")
        self.pause_btn.pack(side="left", padx=(6, 0))
        self.rewind_btn.pack(side="left", padx=(6, 0))

        ttk.Checkbutton(controls, text="Loop", variable=self.loop_var).pack(side="left", padx=(12, 0))

        ttk.Checkbutton(
            controls,
            text="Bones",
            variable=self.bones_var,
            command=self._on_toggle_bones,
        ).pack(side="left", padx=(12, 0))

        ttk.Button(controls, text="Geosets", command=self._open_geosets_popup).pack(side="left", padx=(12, 0))

        ttk.Label(controls, text="Player").pack(side="left", padx=(12, 0))

        self.player_spin = ttk.Spinbox(
            controls,
            from_=0,
            to=11,
            width=3,
            textvariable=self.player_var,
            command=self._on_player_change,
        )
        self.player_spin.pack(side="left", padx=(4, 0))

        # also handle typing + enter
        self.player_spin.bind("<Return>", lambda _e: self._on_player_change())
        self.player_spin.bind("<FocusOut>", lambda _e: self._on_player_change())

        self.time_lbl = ttk.Label(controls, text="t=0ms")
        self.time_lbl.pack(side="right")

        # Ctrl+R camera reset (bind to this window; avoid bind_all)
        self.bind("<Control-r>", lambda _e: self._reset_camera())
        self.bind("<Control-R>", lambda _e: self._reset_camera())

        # load and render first frame immediately
        self._load_from_db()
        self._render_current()

        self.protocol("WM_DELETE_WINDOW", self._on_close)


    def _open_geosets_popup(self) -> None:
        # Lazily build a popup with checkboxes for each geoset in the current mesh.
        if self._mesh is None:
            return
        sub = getattr(self._mesh, "submeshes", None)
        count = len(sub) if sub else 1

        if self._geosets_popup is not None and self._geosets_popup.winfo_exists():
            try:
                self._geosets_popup.lift()
            except Exception:
                pass
            return

        top = tk.Toplevel(self)
        top.title("Geosets")
        top.resizable(False, True)
        self._geosets_popup = top

        # Ensure vars exist and default ON
        self._geoset_vars = []
        for i in range(count):
            v = tk.BooleanVar(value=True)
            self._geoset_vars.append(v)

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        # All / None buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="All", command=lambda: self._set_all_geosets(True)).pack(side="left")
        ttk.Button(btns, text="None", command=lambda: self._set_all_geosets(False)).pack(side="left", padx=(6,0))

        for i, v in enumerate(self._geoset_vars):
            cb = ttk.Checkbutton(frm, text=f"Geoset {i}", variable=v, command=self._on_geoset_toggle)
            cb.pack(anchor="w")

        top.protocol("WM_DELETE_WINDOW", lambda: top.destroy())

    def _on_debug_flags_changed(self) -> None:
        """Push debug flags into GL widget and redraw."""
        try:
            if self.gl is None:
                return

            # existing flags
            try:
                self.gl.set_debug_alpha_off(bool(self.dbg_alpha_off_var.get()))
            except Exception:
                pass
            try:
                self.gl.set_debug_disable_textures(bool(self.dbg_disable_textures_var.get()))
            except Exception:
                pass
            try:
                self.gl.set_debug_color_by_tri(bool(self.dbg_color_by_tri_var.get()))
            except Exception:
                pass
            try:
                self.gl.set_debug_enabled(bool(self.dbg_prints_var.get()))
            except Exception:
                pass

            # NEW: UV debug
            try:
                self.gl.set_debug_flip_v(bool(self.dbg_flip_v_var.get()))
            except Exception:
                pass

            # NEW: TeamColor debug
            try:
                self.gl.set_teamcolor_mode(str(self.teamcolor_mode_var.get()))
            except Exception:
                pass
            try:
                self.gl.set_teamcolor_blend(str(self.teamcolor_blend_var.get()))
            except Exception:
                pass

        except Exception:
            pass

    def _set_all_geosets(self, val: bool) -> None:
        for v in self._geoset_vars:
            try:
                v.set(bool(val))
            except Exception:
                pass
        self._on_geoset_toggle()

    def _on_geoset_toggle(self) -> None:
        try:
            enabled = [bool(v.get()) for v in self._geoset_vars]
        except Exception:
            enabled = None
        if self.gl is not None:
            try:
                self.gl.set_enabled_geosets(enabled)
            except Exception:
                pass

    def _on_player_change(self) -> None:
        try:
            if self.gl is not None:
                self.gl.set_player_index(int(self.player_var.get()))
        except Exception:
            pass

    def _on_toggle_bones(self) -> None:
        try:
            if self.gl is not None:
                self.gl.set_show_bones(bool(self.bones_var.get()))
        except Exception:
            pass

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
        self._seq_start_ms = int(seq.start_ms)
        self._seq_end_ms = int(seq.end_ms)
        self._seq_dur_ms = self._seq_end_ms - self._seq_start_ms
        
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

        print(f"[viewer] MDL path from bones_json['mdl'] = {mdl_path}")
        print(f"[viewer] MDL exists={mdl_path.exists()} size={mdl_path.stat().st_size}")

        # ---------------------------------------------------------------------
        # REAL MESH LOAD: parse MDL from disk (DB mesh not implemented yet)
        # ---------------------------------------------------------------------
        try:
            from .mesh_provider import MdlFileMeshProvider

            self._mesh_provider = MdlFileMeshProvider(mdl_path=mdl_path)
            self._mesh = self._mesh_provider.load_mesh(con=self.con, unit_id=self.unit_id)

            if self._mesh is None:
                print("[viewer] Mesh provider returned None (bones-only)")
            else:
                self.gl.set_mesh(self._mesh)

                m = self._mesh
                sub_ct = len(m.submeshes) if getattr(m, "submeshes", None) else 0
                print(
                    "[viewer] Mesh loaded:"
                    f" verts={len(m.vertices)} tris={len(m.triangles)}"
                    f" submeshes={sub_ct}"
                    f" vgroups={'yes' if m.vertex_groups else 'no'}"
                    f" groups_matrices={'yes' if m.groups_matrices else 'no'}"
                )
                print(
                    f"[viewer] mesh extras:"
                    f" uvs={'None' if getattr(m,'uvs',None) is None else len(getattr(m,'uvs'))}"
                    f" texture_name={getattr(m,'texture_name',None)!r}"
                    f" textures_ct={0 if getattr(m,'textures',None) is None else len(m.textures)}"
                    f" materials_ct={0 if getattr(m,'materials',None) is None else len(m.materials)}"
                )

                # If wrapper has submeshes, print a quick per-geoset summary
                if sub_ct:
                    for i, sm in enumerate(m.submeshes[:10]):  # cap spam
                        print(
                            f"[viewer]   geoset[{i}] verts={len(sm.vertices)} tris={len(sm.triangles)}"
                            f" mat_id={getattr(sm,'geoset_material_id',None)}"
                            f" uvs={'None' if getattr(sm,'uvs',None) is None else len(sm.uvs)}"
                            f" vgroups={'yes' if sm.vertex_groups else 'no'}"
                            f" gmat={'yes' if sm.groups_matrices else 'no'}"
                        )
                    if sub_ct > 10:
                        print(f"[viewer]   ... {sub_ct-10} more geosets")

        except Exception as e:
            self._mesh = None
            print(f"[viewer] Mesh load failed (bones-only): {e!r}")

        # --- Build Nodes and Rig ---
        print("Building Nodes and Rig")
        nodes_by_id = load_nodes_from_mdl(mdl_path)
        rig = build_rig_from_mdl_nodes(nodes_by_id)

        anims = build_anims_from_mdl(mdl_path)

        # --- DEBUG: do boneanim key times line up with sequence times? ---
        self._time_offset = 0
        print("Building All Times")
        all_times: list[int] = []
        try:
            for ch in anims.values():
                all_times.extend([int(k.time_ms) for k in ch.translation])
                all_times.extend([int(k.time_ms) for k in ch.rotation])
                all_times.extend([int(k.time_ms) for k in ch.scaling])
        except Exception:
            all_times = []

        try:
            seq_start = int(self.seq.start_ms)
            seq_end = int(self.seq.end_ms)
            seq_dur = int(seq_end - seq_start)

            times_in_abs = [t for t in all_times if seq_start <= t <= seq_end]
            times_in_rel = [t for t in all_times if 0 <= t <= seq_dur]

            # If there are no keys in the absolute window for this sequence, but there ARE keys
            # in the relative window, treat keys as relative and subtract the sequence start.
            if (not times_in_abs) and times_in_rel and seq_start != 0:
                self._time_offset = seq_start
                print(f"[viewer] using RELATIVE key times for this sequence; offset={self._time_offset}ms")
            else:
                self._time_offset = 0
                if times_in_abs:
                    print(f"[viewer] using ABSOLUTE key times for this sequence; keys_in_window={len(times_in_abs)}")
            self._time_offset = 0
            print(
                f"[viewer] seq={self.seq.name} start={seq_start} end={seq_end} dur={seq_dur}"
            )
            if all_times:
                print(
                    f"[viewer] key_times: min={min(all_times)} max={max(all_times)} count={len(all_times)} "
                    f"| in_abs={len(times_in_abs)} in_rel={len(times_in_rel)}"
                )
            else:
                print("[viewer] key_times: EMPTY (no animation keys parsed)")
        except Exception as e:
            print(f"[viewer] time-domain detection failed: {e!r}")


        self._rig = rig
        self._evaluator = UnitAnimEvaluator(rig=rig, anims=anims)

        # Store seq window for per-channel sampling
        self._seq_start_ms = int(self.seq.start_ms)
        self._seq_end_ms = int(self.seq.end_ms)
        self._seq_dur_ms = int(self._seq_end_ms - self._seq_start_ms)

        # Use a stable time for bind pose; avoid (0,0,0)
        bind_pose = self._evaluator.evaluate_pose(self._seq_start_ms, self._seq_start_ms, self._seq_dur_ms)
        self.gl.set_bind_pose(bind_pose)



    def _render_current(self) -> None:
        if self._evaluator is None or self._rig is None or self.seq is None:
            return

        # Use absolute timeline time; evaluator will decide per-channel whether to use abs or rel.
        t_abs = int(self.t_ms)
        seq_start = int(self.seq.start_ms)
        seq_dur = int(self.seq.end_ms - self.seq.start_ms)

        pose = self._evaluator.evaluate_pose(t_abs, seq_start, seq_dur)

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

        self.time_lbl.config(text=f"t={t_abs}ms")
