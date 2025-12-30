#wc3kin/viewer/window.py
from __future__ import annotations

import tkinter as tk
from PIL import Image
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable

import sqlite3
import os
import re
import shutil
import subprocess
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

from wc3kin.wc3mdl.import_mdl import import_mdl

from wc3kin.viewer.evaluator import build_rig_from_imported_model, build_anims_for_sequence
from wc3kin.wc3mdl import query as mdlq
from .evaluator import Pose, mat4_identity, transform_point


class ViewerPanel(ttk.Frame):
    """
    Embedded viewer panel:
      - OpenGL widget inside a Frame (no separate window)
      - play/pause/rewind
      - loop toggle
      - timeline scrub (quantized w/ shift/ctrl)
      - playback speed multiplier

    If you still want a toplevel window version, use ViewerWindow (wrapper).
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
        on_close: Optional[Callable[[], None]] = None,
        build_menus_on: Optional[tk.Misc] = None,
    ) -> None:
        super().__init__(master)

        self._on_close_cb = on_close

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

        # view bones off by default
        self.bones_var = tk.BooleanVar(value=False)

        # number of player for team color
        self.player_var = tk.IntVar(value=0)
        self._geoset_vars = []  # list[tk.BooleanVar]
        self._geosets_popup = None

        self.playing = False
        # loop off by default
        self.loop_var = tk.BooleanVar(value=False)

        self.seq: Optional[SequenceDef] = None
        self.t_ms: int = 0

        # ---- animation UI state ----
        # timeline_var is sequence-relative time (ms): 0..duration
        self.timeline_var = tk.DoubleVar(value=0.0)
        # playback speed multiplier applied only while playing
        self.speed_var = tk.DoubleVar(value=1.0)

        # Internal guards for timeline scrubbing
        self._scrubbing = False
        self._was_playing_before_scrub = False
        self._ignore_timeline_callback = False

        # Optional debug prints for animation timing / quantization
        self.debug_anim_var = tk.BooleanVar(value=False)

        self._evaluator: Optional[UnitAnimEvaluator] = None
        self._rig = None

        # persistence (camera)
        self._persist = ViewerPersist.load(default_persistence_path(self.con))
        self._cam_init_done = False

        # layout (this Frame is the root)
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
        timeline = ttk.Frame(top)
        timeline.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Label(timeline, text="Timeline").pack(side="left")

        # tk.Scale gives us proper mouse events + modifier state for quantized scrubbing
        self.timeline_scale = tk.Scale(
            timeline,
            from_=0,
            to=1000,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            variable=self.timeline_var,
            length=420,
        )
        self.timeline_scale.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Bind scrub interactions (press/drag/release + track clicks)
        self.timeline_scale.bind("<Button-1>", self._on_timeline_press, add=True)
        self.timeline_scale.bind("<B1-Motion>", self._on_timeline_drag, add=True)
        self.timeline_scale.bind("<ButtonRelease-1>", self._on_timeline_release, add=True)

        # Controls
        self.play_btn = ttk.Button(controls, text="Play", command=self._on_play)
        self.pause_btn = ttk.Button(controls, text="Pause", command=self._on_pause)
        self.rewind_btn = ttk.Button(controls, text="Rewind", command=self._on_rewind)
        self.export_btn = ttk.Button(controls, text="Export…", command=self._on_export)

        self.play_btn.pack(side="left")
        self.pause_btn.pack(side="left", padx=(6, 0))
        self.rewind_btn.pack(side="left", padx=(6, 0))
        self.export_btn.pack(side="left", padx=(6, 0))

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

        # Playback speed (applies only while playing)
        speed_frame = ttk.Frame(controls)
        speed_frame.pack(side="right", padx=(12, 0))
        ttk.Label(speed_frame, text="Speed").pack(side="left")

        self.speed_scale = ttk.Scale(
            speed_frame,
            from_=0.10,
            to=3.00,
            orient="horizontal",
            length=120,
            variable=self.speed_var,
            command=self._on_speed_changed,
        )
        self.speed_scale.pack(side="left", padx=(6, 0))
        self.speed_val_lbl = ttk.Label(speed_frame, text="1.00x", width=6)
        self.speed_val_lbl.pack(side="left", padx=(6, 0))

        self._on_speed_changed()

        self.time_lbl = ttk.Label(controls, text="t=0ms")
        self.time_lbl.pack(side="right")

        # Optional menubar: only do this if caller provides a window-like target
        if build_menus_on is not None:
            self._build_menus(build_menus_on)

        # load and render first frame immediately
        self._load_from_db()
        self._render_current()
    ##----------------------------------------
    ##-- Helper Functions
    ##----------------------------------------
    def load_unit_sequence(self, unit_id: int, sequence_name: str) -> None:
        """
        Reuse the existing GL widget and reload model/sequence.
        Safe for embedding: does not destroy/recreate the OpenGLFrame.
        """
        # stop playback during reload
        self.playing = False
        self._scrubbing = False
        self._was_playing_before_scrub = False

        unit_id = int(unit_id)
        sequence_name = str(sequence_name)

        # no-op if nothing changed
        if getattr(self, "unit_id", None) == unit_id and getattr(self, "sequence_name", None) == sequence_name:
            return

        # update selection
        self.unit_id = unit_id
        self.sequence_name = sequence_name

        # reset camera-init so fit/restore runs for new model/seq
        self._cam_init_done = False

        # close any old geoset popup (it references old mesh count)
        try:
            if self._geosets_popup is not None and self._geosets_popup.winfo_exists():
                self._geosets_popup.destroy()
        except Exception:
            pass
        self._geosets_popup = None
        self._geoset_vars = []

        # reload + render first frame
        self._load_from_db()
        self._render_current()


    # -----------------------------
    # Menu / keybind helpers
    # -----------------------------
    def _build_menus(self, target: tk.Misc) -> None:
        # Menus only work properly on a toplevel/root
        menubar = tk.Menu(target)
        try:
            target.config(menu=menubar)
        except Exception:
            return

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)

        debug_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Debug", menu=debug_menu)

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
        debug_menu.add_separator()
        debug_menu.add_checkbutton(label="Anim Debug Prints", variable=self.debug_anim_var)

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

        # Ctrl+R camera reset (bind to target window; avoid bind_all)
        try:
            target.bind("<Control-r>", lambda _e: self._reset_camera())
            target.bind("<Control-R>", lambda _e: self._reset_camera())
        except Exception:
            pass

    # -----------------------------
    # Speed control
    # -----------------------------
    def _on_speed_changed(self, _val: str | float | None = None) -> None:
        """Update the speed label; tick loop reads speed_var every frame."""
        try:
            sp = float(self.speed_var.get() or 1.0)
        except Exception:
            sp = 1.0
        sp = max(0.10, min(3.00, sp))
        if abs(sp - float(self.speed_var.get() or 1.0)) > 1e-6:
            self.speed_var.set(sp)
        if hasattr(self, "speed_val_lbl"):
            self.speed_val_lbl.config(text=f"{sp:.2f}x")
        if self.debug_anim_var.get():
            print(f"[anim] speed set to {sp:.3f}x")

    # -----------------------------
    # Timeline quantization / scrubbing
    # -----------------------------
    @staticmethod
    def _mod_quant_step_from_state(state: int) -> int:
        """Return scrub quantization step (ms) based on modifier keys."""
        SHIFT_MASK = 0x0001
        CTRL_MASK = 0x0004
        if state & CTRL_MASK:
            return 1
        if state & SHIFT_MASK:
            return 10
        return 33

    def _quantize_t_rel(self, raw_t_rel: float, *, state: int) -> tuple[int, int]:
        """Quantize raw sequence-relative time to avoid jitter while scrubbing."""
        dur = int(getattr(self, "_seq_dur_ms", 0) or 0)
        step = self._mod_quant_step_from_state(state)
        if dur <= 0:
            return 0, step
        q = int(round(float(raw_t_rel) / float(step)) * step)
        q = max(0, min(dur, q))
        return q, step

    def _set_time_from_t_rel(self, t_rel_ms: int) -> None:
        """Set absolute cursor (t_ms) from a sequence-relative time and redraw."""
        if not hasattr(self, "_seq_start_ms"):
            return
        seq_start = int(self._seq_start_ms)
        dur = int(getattr(self, "_seq_dur_ms", 0) or 0)
        t_rel_ms = max(0, min(dur, int(t_rel_ms)))
        self.t_ms = float(seq_start + t_rel_ms)
        self._render_current()

    def _update_timeline_bounds(self) -> None:
        """Update slider range from ImportedModel sequence timing."""
        dur = int(getattr(self, "_seq_dur_ms", 0) or 0)
        if hasattr(self, "timeline_scale"):
            try:
                self.timeline_scale.config(to=max(1, dur))
            except Exception:
                pass
        self._ignore_timeline_callback = True
        try:
            cur = float(self.timeline_var.get() or 0.0)
            self.timeline_var.set(max(0.0, min(float(dur), cur)))
        finally:
            self._ignore_timeline_callback = False

    def _on_timeline_press(self, event: tk.Event) -> None:
        self._scrubbing = True
        self._was_playing_before_scrub = bool(self.playing)
        self.playing = False

        def apply_click_quant() -> None:
            if self._ignore_timeline_callback:
                return
            raw = float(self.timeline_var.get() or 0.0)
            q, step = self._quantize_t_rel(raw, state=int(getattr(event, "state", 0) or 0))
            self._ignore_timeline_callback = True
            try:
                self.timeline_var.set(float(q))
            finally:
                self._ignore_timeline_callback = False
            if self.debug_anim_var.get():
                print(f"[anim] scrub press raw={raw:.2f} q={q} step={step}ms")
            self._set_time_from_t_rel(q)

        self.after_idle(apply_click_quant)

    def _on_timeline_drag(self, event: tk.Event) -> None:
        if self._ignore_timeline_callback:
            return
        raw = float(self.timeline_var.get() or 0.0)
        q, step = self._quantize_t_rel(raw, state=int(getattr(event, "state", 0) or 0))
        self._ignore_timeline_callback = True
        try:
            self.timeline_var.set(float(q))
        finally:
            self._ignore_timeline_callback = False
        if self.debug_anim_var.get():
            print(f"[anim] scrub drag raw={raw:.2f} q={q} step={step}ms")
        self._set_time_from_t_rel(q)

    def _on_timeline_release(self, event: tk.Event) -> None:
        if self._ignore_timeline_callback:
            self._scrubbing = False
            return
        raw = float(self.timeline_var.get() or 0.0)
        q, step = self._quantize_t_rel(raw, state=int(getattr(event, "state", 0) or 0))
        self._ignore_timeline_callback = True
        try:
            self.timeline_var.set(float(q))
        finally:
            self._ignore_timeline_callback = False
        self._set_time_from_t_rel(q)

        # resume if we were playing before scrubbing
        self.playing = bool(self._was_playing_before_scrub)

        if self.debug_anim_var.get():
            print(f"[anim] scrub release raw={raw:.2f} q={q} step={step}ms resume={self.playing}")

        self._scrubbing = False
        if self.playing:
            # ensure tick loop continues after scrub
            self.after(self.TICK_MS, self._tick)

    # -----------------------------
    # Misc UI actions
    # -----------------------------
    def _open_geosets_popup(self) -> None:
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

        self._geoset_vars = []
        for _i in range(count):
            v = tk.BooleanVar(value=True)
            self._geoset_vars.append(v)

        frm = ttk.Frame(top, padding=10)
        frm.pack(fill="both", expand=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="All", command=lambda: self._set_all_geosets(True)).pack(side="left")
        ttk.Button(btns, text="None", command=lambda: self._set_all_geosets(False)).pack(side="left", padx=(6, 0))

        for i, v in enumerate(self._geoset_vars):
            cb = ttk.Checkbutton(frm, text=f"Geoset {i}", variable=v, command=self._on_geoset_toggle)
            cb.pack(anchor="w")

        top.protocol("WM_DELETE_WINDOW", lambda: top.destroy())

    def _on_debug_flags_changed(self) -> None:
        """Push debug flags into GL widget and redraw."""
        try:
            if self.gl is None:
                return

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

            try:
                self.gl.set_debug_flip_v(bool(self.dbg_flip_v_var.get()))
            except Exception:
                pass

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

    def _make_neutral_bind_pose(self):
        world_mats = {}
        for nid in self._rig.ids:
            world_mats[nid] = mat4_identity()

        max_id = max(self._rig.ids) if self._rig.ids else -1
        world_pos = [(0.0, 0.0, 0.0)] * (max_id + 1 if max_id >= 0 else 0)
        for nid in self._rig.ids:
            pv = self._rig.pivot.get(nid, (0.0, 0.0, 0.0))
            if nid < len(world_pos):
                world_pos[nid] = transform_point(world_mats[nid], pv)

        return Pose(world_mats=world_mats, world_pos=world_pos)

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

    def close(self) -> None:
        """Stop playback, persist camera, and run optional close callback."""
        self.playing = False

        try:
            cam = self.gl.get_camera_state() if self.gl is not None else None
            if cam:
                self._persist.set_camera(self.unit_id, self.sequence_name, cam)
                self._persist.save()
        except Exception:
            pass

        if self._on_close_cb:
            try:
                self._on_close_cb()
            except Exception:
                pass

    # -----------------------------
    # Playback controls
    # -----------------------------
    def _on_play(self) -> None:
        if self._evaluator is None:
            print("Evaluator is None")
            return

        if not hasattr(self, "_seq_start_ms") or not hasattr(self, "_seq_end_ms"):
            print("No Start and / or Stop cannot animate")
            return

        if not self.playing:
            self.playing = True
            self._tick()

    def _on_pause(self) -> None:
        self.playing = False

    def _on_rewind(self) -> None:
        if not hasattr(self, "_seq_start_ms"):
            return
        self.t_ms = int(self._seq_start_ms)
        self._render_current()

    def _tick(self) -> None:
        if not self.playing or self._evaluator is None:
            return

        if not hasattr(self, "_seq_start_ms") or not hasattr(self, "_seq_end_ms"):
            return

        speed = float(self.speed_var.get() or 1.0)
        self.t_ms += self.TICK_MS * speed
        if self.debug_anim_var.get():
            print(f"[anim] tick speed={speed:.3f} t_abs={int(self.t_ms)}")

        seq_start = int(self._seq_start_ms)
        seq_end = int(self._seq_end_ms)

        if self.t_ms > seq_end:
            if self.loop_var.get():
                self.t_ms = seq_start
            else:
                self.t_ms = seq_start
                self.playing = False

        self._render_current()

        if self.playing:
            self.after(self.TICK_MS, self._tick)

    # -----------------------------
    # Data loading + render pipeline (unchanged)
    # -----------------------------
    def _load_from_db(self) -> None:
        print("load from db called")
        self._mesh = None
        self._mesh_provider = None
        self._mdl = None  # ImportedModel (wc3mdl) stored for alpha/geosets/etc.

        bones_json = dbmod.get_harvested_json_blob(self.con, self.unit_id, "bones")
        if bones_json is None:
            raise RuntimeError(
                "Missing harvested JSON blob in DB: kind='bones'.\n"
                "Expected bones blob to contain at least {'mdl': <path>}."
            )

        mdl_path_str = bones_json.get("mdl")
        if not mdl_path_str:
            print("bones_json missing 'mdl' path; cannot load MDL for viewer.")
            raise RuntimeError("bones_json missing 'mdl' path; cannot load MDL for viewer.")
        mdl_path = Path(mdl_path_str)

        if not mdl_path.exists():
            print(f"MDL path does not exist on disk: {mdl_path}")
            raise RuntimeError(f"MDL path does not exist on disk: {mdl_path}")

        print(f"[viewer] MDL path from bones_json['mdl'] = {mdl_path}")
        print(f"[viewer] MDL exists={mdl_path.exists()} size={mdl_path.stat().st_size}")

        # 1) Load mesh from disk (unchanged)
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

        print("[viewer] importing MDL via wc3mdl.import_mdl ...")
        self._mdl = import_mdl(str(mdl_path))
        self._mdl_path=mdl_path
        seq = next((s for s in self._mdl.sequences if s.name == self.sequence_name), None)
        if seq is None:
            available = [s.name for s in self._mdl.sequences]
            raise RuntimeError(
                f"Sequence not found in imported model: {self.sequence_name!r}\n"
                f"Available sequences: {available}"
            )

        # Persist the selected sequence for features (e.g., export) that need
        # access to the imported-model sequence interval.
        self._seq = seq

        self._seq_start_ms = int(seq.start_abs)
        self._seq_end_ms = int(seq.end_abs)
        self._seq_dur_ms = int(seq.dur)
        self.t_ms = int(seq.start_abs)
        self._update_timeline_bounds()

        print(f"[viewer] seq={seq.name!r} start={self._seq_start_ms} end={self._seq_end_ms} dur={self._seq_dur_ms}")

        rig = build_rig_from_imported_model(self._mdl)

        play_anims = build_anims_for_sequence(self._mdl, seq.name)
        self._rig = rig
        self._evaluator = UnitAnimEvaluator(rig=rig, anims=play_anims)

        bind_seq = next((s for s in self._mdl.sequences if s.name == "Stand"), None)
        if bind_seq is None:
            bind_seq = seq

        bind_anims = build_anims_for_sequence(self._mdl, bind_seq.name)
        bind_eval = UnitAnimEvaluator(rig=rig, anims=bind_anims)

        bind_start = int(bind_seq.start_abs)
        bind_dur = int(bind_seq.dur)

        # bind_pose = bind_eval.evaluate_pose(bind_start, bind_start, bind_dur)
        bind_pose = self._make_neutral_bind_pose()
        self.gl.set_bind_pose(bind_pose)

        print(f"[viewer] bind_pose set from {bind_seq.name!r} at t_abs={bind_start} (dur={bind_dur})")

        self.t_ms = int(seq.start_abs)
        self._render_current()

        try:
            ga_ct = len(getattr(self._mdl, "geoset_anims", {}) or {})
            gs_ct = len(getattr(self._mdl, "geosets", []) or [])
            print(f"[viewer] ImportedModel geosets={gs_ct} geoset_anims={ga_ct}")
        except Exception:
            pass

        rig_ids = set(self._rig.ids)

        def _collect_influence_ids(mesh) -> set[int]:
            ids: set[int] = set()
            if mesh is None:
                return ids

            def add_from_vgroups(vgroups):
                if not vgroups:
                    return

                if isinstance(vgroups, (list, tuple)) and vgroups and isinstance(vgroups[0], int):
                    ids.update(int(x) for x in vgroups)
                    return

                for infl_list in vgroups:
                    if infl_list is None:
                        continue

                    if isinstance(infl_list, int):
                        ids.add(int(infl_list))
                        continue

                    if isinstance(infl_list, dict):
                        for k in ("id", "bone", "bone_id", "group", "matrix"):
                            if k in infl_list:
                                try:
                                    ids.add(int(infl_list[k]))
                                except Exception:
                                    pass
                                break
                        continue

                    if isinstance(infl_list, tuple):
                        if len(infl_list) >= 1:
                            try:
                                ids.add(int(infl_list[0]))
                            except Exception:
                                pass
                        continue

                    if isinstance(infl_list, (list, tuple)):
                        for infl in infl_list:
                            if isinstance(infl, int):
                                ids.add(int(infl))
                            elif isinstance(infl, tuple) and len(infl) >= 1:
                                try:
                                    ids.add(int(infl[0]))
                                except Exception:
                                    pass
                            elif isinstance(infl, dict):
                                for k in ("id", "bone", "bone_id", "group", "matrix"):
                                    if k in infl:
                                        try:
                                            ids.add(int(infl[k]))
                                        except Exception:
                                            pass
                                        break

            add_from_vgroups(getattr(mesh, "vertex_groups", None))
            for sm in (getattr(mesh, "submeshes", None) or []):
                add_from_vgroups(getattr(sm, "vertex_groups", None))

            return ids

        inf_ids = _collect_influence_ids(self._mesh)

        print("[dbg] influence ids:", len(inf_ids))
        print("[dbg] influence not in rig:", sorted(list(inf_ids - rig_ids))[:50])
        print("[dbg] rig not referenced by influence:", sorted(list(rig_ids - inf_ids))[:50])

    def _render_current(self) -> None:
        if self._evaluator is None or self._rig is None or self._mdl is None:
            return

        t_abs = int(self.t_ms)

        seq_start = int(self._seq_start_ms)
        seq_dur = int(self._seq_dur_ms)

        pose = self._evaluator.evaluate_pose(t_abs, seq_start, seq_dur)

        enabled = []
        geosets = getattr(self._mdl, "geosets", []) or []
        geoset_anims = getattr(self._mdl, "geoset_anims", {}) or {}

        for gid in range(len(geosets)):
            user_on = True
            if self._geoset_vars:
                try:
                    user_on = bool(self._geoset_vars[gid].get())
                except Exception:
                    user_on = True

            ga = geoset_anims.get(gid)
            a = 1.0 if ga is None else mdlq.eval_geoset_alpha(ga, t_abs)

            enabled.append(user_on and (a > 0.01))

        self.gl.set_enabled_geosets(enabled)

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

            self.gl.snapshot_default_camera()
            self._cam_init_done = True

        self.gl.set_pose(pose, self._rig, active_ids=None)

        t_rel = int(t_abs - seq_start)
        pct = 0.0 if seq_dur <= 0 else (max(0.0, min(1.0, t_rel / float(seq_dur))) * 100.0)

        active_seq_name = getattr(getattr(self, "_evaluator", None), "sequence_name", None)
        if not active_seq_name:
            active_seq_name = getattr(self, "sequence_name", "(seq)")

        self.time_lbl.config(text=f"{active_seq_name}  {t_rel}ms  ({pct:.1f}%)")

        if not self._scrubbing:
            self._ignore_timeline_callback = True
            try:
                self.timeline_var.set(float(t_rel))
            finally:
                self._ignore_timeline_callback = False

        if self.debug_anim_var.get():
            print(f"[anim] seq={active_seq_name!r} dur={seq_dur} t_rel={t_rel} t_abs={t_abs}")


    # -----------------------------
    # Export UI + implementation
    # -----------------------------

    def _on_export(self) -> None:
        if not getattr(self, "_seq", None):
            messagebox.showinfo("Export", "No animation is selected.")
            return
        self.ExportDialog(self)

    def _derive_default_export_name(self, fmt: str) -> str:
        fmt = (fmt or "gif").lower()
        seq = getattr(self, "_seq", None)
        anim_name = getattr(seq, "name", "animation") if seq else "animation"
        anim_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", anim_name.strip()) or "animation"

        mdl_path = None
        try:
            mdl_path = Path(getattr(self, "_mdl_path", "")) if getattr(self, "_mdl_path", None) else None
        except Exception:
            mdl_path = None

        race = "unknown"
        unit = "unit"
        if mdl_path and mdl_path.exists():
            parts = [p.lower() for p in mdl_path.parts]
            for r in ("human", "orc", "undead", "nightelf", "neutral", "naga", "demon"):
                if r in parts:
                    race = r
                    break
            unit = mdl_path.parent.name or mdl_path.stem
        else:
            # best-effort fallback using window title / selector
            print(f'Could not determine unit info from path:{mdl_path}')
            try:
                unit = getattr(self, "_unit_name", "unit")
            except Exception:
                unit = "unit"

        unit_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(unit).strip()) or "unit"
        race_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(race).strip()) or "unknown"
        return f"{race_slug}-{unit_slug}-{anim_slug}.{fmt}"

    def _find_ffmpeg(self, user_path: str = "") -> Optional[str]:
        p = (user_path or "").strip().strip('"')
        if p:
            if os.path.isdir(p):
                cand = os.path.join(p, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
                if os.path.exists(cand):
                    return cand
            if os.path.exists(p):
                return p

        cand = shutil.which("ffmpeg")
        if cand:
            return cand

        # Common fallback locations
        candidates = []
        if os.name == "nt":
            candidates += [
                r"C:\\ffmpeg\\bin\\ffmpeg.exe",
                r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                r"C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
            ]
        else:
            candidates += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _export_animation(
        self,
        out_path: str,
        fmt: str,
        start_ms: float,
        end_ms: float,
        out_w: Optional[int],
        out_h: Optional[int],
        fps: int,
        ffmpeg_path: str = "",
    ) -> None:
        """Export frames by stepping evaluator across time and capturing OpenGL framebuffer."""
        fmt = (fmt or "gif").lower().strip()
        if fmt not in ("gif", "mp4"):
            raise ValueError("Unsupported format")

        if self._evaluator is None or self._rig is None:
            raise RuntimeError("Viewer is not ready (missing evaluator/rig)")

        seq = self._seq
        if seq is None:
            raise RuntimeError("No animation selected")

        # SequenceClip stores absolute timeline bounds as start_abs/end_abs and
        # duration as dur.
        seq_start = float(getattr(seq, "start_abs", 0.0))
        seq_end = float(getattr(seq, "end_abs", seq_start + float(getattr(seq, "dur", 1.0))))
        seq_dur = float(max(1.0, float(getattr(seq, "dur", max(1.0, seq_end - seq_start)))))

        # clamp/normalize to relative ms within the sequence
        s = max(0.0, min(float(start_ms), seq_dur))
        e = max(0.0, min(float(end_ms), seq_dur))
        if e <= s:
            raise ValueError("End must be greater than start")

        fps = int(fps) if int(fps) > 0 else 30

        # Honor the viewer's playback speed multiplier so exported motion matches
        # what the user sees when replay speed is adjusted.
        try:
            speed_mul = float(self.speed_var.get() or 1.0)
        except Exception:
            speed_mul = 1.0
        if speed_mul <= 0:
            speed_mul = 1.0

        # Time step in *animation milliseconds* per output frame.
        step = (1000.0 / float(fps)) * speed_mul

        images = []
        t = s
        # Determine active bone IDs for rendering (matches _render_current behavior)
        active_ids = None
        try:
            active_ids = getattr(self, "_active_bone_ids", None)
        except Exception:
            active_ids = None

        while t <= e + 0.0001:
            t_abs = seq_start + t
            pose = self._evaluator.evaluate_pose(t_abs, seq_start, seq_dur)
            self.gl.set_pose(pose, self._rig, active_ids)
            # Force a draw (this will freeze UI; user said that's OK)
            self.update_idletasks()
            self.update()
            img = self.gl.capture_image()
            if img is None:
                raise RuntimeError("OpenGL capture is unavailable (missing pyopengltk/PyOpenGL?)")

            if out_w or out_h:
                w0, h0 = img.size
                if out_w and out_h:
                    img = img.resize((int(out_w), int(out_h)), resample=Image.Resampling.LANCZOS)
                elif out_w:
                    nh = int(round(h0 * (float(out_w) / float(w0))))
                    img = img.resize((int(out_w), nh), resample=Image.Resampling.LANCZOS)
                elif out_h:
                    nw = int(round(w0 * (float(out_h) / float(h0))))
                    img = img.resize((nw, int(out_h)), resample=Image.Resampling.LANCZOS)

            images.append(img)
            t += step

        if not images:
            raise RuntimeError("No frames captured")

        out_path = str(out_path)

        if fmt == "gif":
            duration_ms = int(round(1000.0 / float(fps)))
            images[0].save(
                out_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            return

        # MP4: write PNG sequence then call ffmpeg
        ffmpeg = self._find_ffmpeg(ffmpeg_path)
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found. Provide a path to ffmpeg to export MP4.")

        import tempfile
        with tempfile.TemporaryDirectory(prefix="wc3kin_export_") as td:
            for i, img in enumerate(images):
                img.save(os.path.join(td, f"frame_{i:05d}.png"))
            # -pix_fmt yuv420p improves compatibility
            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(td, "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "ffmpeg failed:\n"
                    + (proc.stdout or "")
                    + "\n"
                    + (proc.stderr or "")
                )


    class ExportDialog(tk.Toplevel):
        def __init__(self, viewer: "ViewerPanel"):
            super().__init__(viewer)
            self.viewer = viewer
            self.title("Export Animation")
            self.resizable(False, False)
            self.transient(viewer.winfo_toplevel())
            self.grab_set()

            self.fmt_var = tk.StringVar(value="gif")
            self.path_var = tk.StringVar(value="")
            self.ffmpeg_var = tk.StringVar(value="")
            self.w_var = tk.StringVar(value="")
            self.h_var = tk.StringVar(value="")
            self.fps_var = tk.StringVar(value="30")
            self._mdl_path=''

            seq = viewer._seq
            seq_start = float(getattr(seq, 'start_abs', 0))
            seq_end = float(getattr(seq, 'end_abs', seq_start + getattr(seq, 'dur', 1)))
            seq_dur = float(max(1, getattr(seq, 'dur', int(seq_end - seq_start))))

            self.start_ms_var = tk.StringVar(value="0")
            self.end_ms_var = tk.StringVar(value=str(int(seq_dur)))

            body = ttk.Frame(self, padding=12)
            body.pack(fill="both", expand=True)

            # Format
            fmt_row = ttk.Frame(body)
            fmt_row.pack(fill="x")
            ttk.Label(fmt_row, text="Format:").pack(side="left")
            ttk.Radiobutton(fmt_row, text="GIF", value="gif", variable=self.fmt_var, command=self._sync_default_name).pack(side="left", padx=(8, 0))
            ttk.Radiobutton(fmt_row, text="MP4", value="mp4", variable=self.fmt_var, command=self._sync_default_name).pack(side="left", padx=(8, 0))

            # Save as
            save_row = ttk.Frame(body)
            save_row.pack(fill="x", pady=(10, 0))
            ttk.Button(save_row, text="Save as…", command=self._choose_out).pack(side="left")
            ttk.Label(save_row, textvariable=self.path_var).pack(side="left", padx=(8, 0))

            # ffmpeg
            ff_row = ttk.Frame(body)
            ff_row.pack(fill="x", pady=(10, 0))
            ttk.Button(ff_row, text="Locate ffmpeg…", command=self._choose_ffmpeg).pack(side="left")
            ttk.Label(ff_row, textvariable=self.ffmpeg_var).pack(side="left", padx=(8, 0))

            # Scaling
            scale = ttk.LabelFrame(body, text="Scaling (optional)")
            scale.pack(fill="x", pady=(10, 0))
            srow = ttk.Frame(scale)
            srow.pack(fill="x", padx=8, pady=6)
            ttk.Label(srow, text="Width:").pack(side="left")
            ttk.Entry(srow, textvariable=self.w_var, width=8).pack(side="left", padx=(6, 12))
            ttk.Label(srow, text="Height:").pack(side="left")
            ttk.Entry(srow, textvariable=self.h_var, width=8).pack(side="left", padx=(6, 0))

            # Trim
            trim = ttk.LabelFrame(body, text="Trim (ms within this animation)")
            trim.pack(fill="x", pady=(10, 0))
            trow = ttk.Frame(trim)
            trow.pack(fill="x", padx=8, pady=6)
            ttk.Label(trow, text="Start:").pack(side="left")
            ttk.Entry(trow, textvariable=self.start_ms_var, width=10).pack(side="left", padx=(6, 12))
            ttk.Label(trow, text="End:").pack(side="left")
            ttk.Entry(trow, textvariable=self.end_ms_var, width=10).pack(side="left", padx=(6, 0))

            # FPS
            fps_row = ttk.Frame(body)
            fps_row.pack(fill="x", pady=(10, 0))
            ttk.Label(fps_row, text="FPS:").pack(side="left")
            ttk.Entry(fps_row, textvariable=self.fps_var, width=6).pack(side="left", padx=(6, 0))

            # Buttons
            btns = ttk.Frame(body)
            btns.pack(fill="x", pady=(12, 0))
            ttk.Button(btns, text="Export", command=self._export).pack(side="right")
            ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

            self._sync_default_name()

        def _sync_default_name(self) -> None:
            fmt = (self.fmt_var.get() or "gif").lower()
            default_name = self.viewer._derive_default_export_name(fmt)

            cur = (self.path_var.get() or "").strip()
            if not cur:
                self.path_var.set(str(Path.cwd() / default_name))
                return

            p = Path(cur)
            # If the user hasn't opened Save As, they’re likely using the auto-name:
            # keep their chosen directory, but sync the extension to the selected format.
            new_path = p.with_suffix(f".{fmt}")
            self.path_var.set(str(new_path))

        def _choose_out(self) -> None:
            fmt = (self.fmt_var.get() or "gif").lower()
            default_name = self.viewer._derive_default_export_name(fmt)
            ext = f".{fmt}"
            path = filedialog.asksaveasfilename(
                parent=self,
                title="Save animation",
                initialfile=default_name,
                defaultextension=ext,
                filetypes=[(fmt.upper(), f"*{ext}"), ("All files", "*.*")],
            )
            if path:
                self.path_var.set(path)

        def _choose_ffmpeg(self) -> None:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select ffmpeg executable",
                filetypes=[("ffmpeg", "ffmpeg*"), ("All files", "*.*")],
            )
            if path:
                self.ffmpeg_var.set(path)

        def _export(self) -> None:
            out_path = (self.path_var.get() or "").strip()
            if not out_path:
                messagebox.showerror("Export", "Choose a Save As path first.")
                return

            fmt = (self.fmt_var.get() or "gif").lower()
            try:
                start_ms = float(self.start_ms_var.get().strip() or "0")
                end_ms = float(self.end_ms_var.get().strip() or "0")
                fps = int(float(self.fps_var.get().strip() or "30"))
                w = self.w_var.get().strip()
                h = self.h_var.get().strip()
                out_w = int(float(w)) if w else None
                out_h = int(float(h)) if h else None
            except Exception:
                messagebox.showerror("Export", "Invalid numeric input (start/end/fps/width/height).")
                return

            try:
                self.viewer._export_animation(
                    out_path=out_path,
                    fmt=fmt,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    out_w=out_w,
                    out_h=out_h,
                    fps=fps,
                    ffmpeg_path=self.ffmpeg_var.get(),
                )
            except Exception as e:
                messagebox.showerror("Export failed", str(e))
                return

            messagebox.showinfo("Export", f"Exported: {out_path}")
            self.destroy()

class ViewerWindow(tk.Toplevel):
    """
    Backwards-compatible wrapper around ViewerPanel.
    Use this if you still want the viewer in its own window.
    """

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

        self._panel = ViewerPanel(
            self,
            con=con,
            units_root=units_root,
            unit_id=unit_id,
            sequence_name=sequence_name,
            on_close=self.destroy,
            build_menus_on=self,
        )
        self._panel.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._panel.close)



