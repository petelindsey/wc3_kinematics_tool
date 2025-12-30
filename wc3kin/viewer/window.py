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

from wc3kin.wc3mdl.import_mdl import import_mdl

from wc3kin.viewer.evaluator import build_rig_from_imported_model, build_anims_for_sequence
from wc3kin.wc3mdl import query as mdlq
from .evaluator import Pose, mat4_identity, transform_point

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
        debug_menu.add_separator()
        debug_menu.add_checkbutton(label="Anim Debug Prints",variable=self.debug_anim_var,)

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

        # Ctrl+R camera reset (bind to this window; avoid bind_all)
        self.bind("<Control-r>", lambda _e: self._reset_camera())
        self.bind("<Control-R>", lambda _e: self._reset_camera())

        # load and render first frame immediately
        self._load_from_db()
        self._render_current()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_speed_changed(self, _val: str | float | None = None) -> None:
        """Update the speed label; tick loop reads speed_var every frame."""
        try:
            sp = float(self.speed_var.get() or 1.0)
        except Exception:
            sp = 1.0
        sp = max(0.10, min(3.00, sp))
        if abs(sp - float(self.speed_var.get() or 1.0)) > 1e-6:
            # Clamp back into var if user typed something odd
            self.speed_var.set(sp)
        if hasattr(self, "speed_val_lbl"):
            self.speed_val_lbl.config(text=f"{sp:.2f}x")
        if self.debug_anim_var.get():
            print(f"[anim] speed set to {sp:.3f}x")

    @staticmethod
    def _mod_quant_step_from_state(state: int) -> int:
        """Return scrub quantization step (ms) based on modifier keys."""
        # Tk state bitmasks are platform-dependent-ish but these are stable on Win/X11.
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
        # Snap to nearest step, then clamp
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
        # Ensure current var is clamped
        self._ignore_timeline_callback = True
        try:
            cur = float(self.timeline_var.get() or 0.0)
            self.timeline_var.set(max(0.0, min(float(dur), cur)))
        finally:
            self._ignore_timeline_callback = False

    def _on_timeline_press(self, event: tk.Event) -> None:
        # Clicking the track should also quantize, so wait until Scale has updated its value.
        self._scrubbing = True
        self._was_playing_before_scrub = bool(self.playing)
        # Scrubbing should never *start* playback; safest is to pause while scrubbing.
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
        # No jitter: keep var exactly on quantized values
        self._ignore_timeline_callback = True
        try:
            self.timeline_var.set(float(q))
        finally:
            self._ignore_timeline_callback = False
        if self.debug_anim_var.get():
            print(f"[anim] scrub drag raw={raw:.2f} q={q} step={step}ms")
        self._set_time_from_t_rel(q)

    def _on_timeline_release(self, event: tk.Event) -> None:
        # Keep current time; do not snap back. (Playback remains paused unless user hits Play.)
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

        # Resume playback if we were playing prior to scrubbing
        self.playing = bool(self._was_playing_before_scrub)

        if self.debug_anim_var.get():
            print(
                f"[anim] scrub release raw={raw:.2f} q={q} step={step}ms "
                f"resume={self.playing}"
            )

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

    def _make_neutral_bind_pose(self):
        world_mats = {}
        # identity for every node id in rig
        for nid in self._rig.ids:
            world_mats[nid] = mat4_identity()

        # optional: positions for bone drawing
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
        if self._evaluator is None:
            print('Evaluator is None')
            return
        
        if not hasattr(self, "_seq_start_ms") or not hasattr(self, "_seq_end_ms"):
        
            print('No Start and / or Stop cannot animate')
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
    
    def _on_rewind(self) -> None:
        if not hasattr(self, "_seq_start_ms"):
            return
        self.t_ms = int(self._seq_start_ms)
        self._render_current()

    def _tick(self) -> None:
        
        if not self.playing or self._evaluator is None:
            print('Self .Playing Not set')
            return

        if self._evaluator is None:
            print('Evaluator is None, Cannot Animate')
            return
        
        # Need a valid sequence window from ImportedModel
        if not hasattr(self, "_seq_start_ms") or not hasattr(self, "_seq_end_ms"):
            print('Start and/or Stop Not Defined, cannot animate')
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
                # stop + auto-rewind
                self.t_ms = seq_start
                self.playing = False

        self._render_current()
        self.after(self.TICK_MS, self._tick)
        
    def _load_from_db(self) -> None:
        print("load from db called")
        self._mesh = None
        self._mesh_provider = None
        self._mdl = None  # ImportedModel (wc3mdl) stored for alpha/geosets/etc.

        # ---------------------------------------------------------------------------------
        # 0) Resolve mdl_path from DB (keep this part since your UI/DB points to mdl)
        # ---------------------------------------------------------------------------------
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

        # ---------------------------------------------------------------------------------
        # 1) Load mesh from disk (unchanged)
        # ---------------------------------------------------------------------------------
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

        # Find sequence by name in the imported model (authoritative)
        seq = next((s for s in self._mdl.sequences if s.name == self.sequence_name), None)
        if seq is None:
            available = [s.name for s in self._mdl.sequences]
            raise RuntimeError(
                f"Sequence not found in imported model: {self.sequence_name!r}\n"
                f"Available sequences: {available}"
            )

        # Store sequence timing (absolute)
        self._seq_start_ms = int(seq.start_abs)
        self._seq_end_ms = int(seq.end_abs)
        self._seq_dur_ms = int(seq.dur)
        self.t_ms = int(seq.start_abs)
        self._update_timeline_bounds()

        print(
            f"[viewer] seq={seq.name!r} start={self._seq_start_ms} end={self._seq_end_ms} dur={self._seq_dur_ms}"
        )

        # --- rig once ---
        rig = build_rig_from_imported_model(self._mdl)

        # --- playback anims (current sequence) ---
        play_anims = build_anims_for_sequence(self._mdl, seq.name)
        self._rig = rig
        self._evaluator = UnitAnimEvaluator(rig=rig, anims=play_anims)

        # --- bind pose anims (prefer Stand) ---
        bind_seq = next((s for s in self._mdl.sequences if s.name == "Stand"), None)
        if bind_seq is None:
            bind_seq = seq  # fallback

        bind_anims = build_anims_for_sequence(self._mdl, bind_seq.name)
        bind_eval = UnitAnimEvaluator(rig=rig, anims=bind_anims)

        bind_start = int(bind_seq.start_abs)
        bind_dur = int(bind_seq.dur)

        #bind_pose = bind_eval.evaluate_pose(bind_start, bind_start, bind_dur)
        bind_pose = self._make_neutral_bind_pose()
        self.gl.set_bind_pose(bind_pose)
        
        

        print(
            f"[viewer] bind_pose set from {bind_seq.name!r} "
            f"at t_abs={bind_start} (dur={bind_dur})"
        )

        # --- start playback cursor at CURRENT seq start ---
        self.t_ms = int(seq.start_abs)
        self._render_current()
        # ---------------------------------------------------------------------------------
        # 5) (Optional) You can print a quick geoset/geosetanim summary here
        # ---------------------------------------------------------------------------------
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

                # Case A: per-vertex single int (very common): [3,3,3,4,4,...]
                if isinstance(vgroups, (list, tuple)) and vgroups and isinstance(vgroups[0], int):
                    ids.update(int(x) for x in vgroups)
                    return

                # Otherwise treat as per-vertex "influence list"
                for infl_list in vgroups:
                    if infl_list is None:
                        continue

                    # Case B: each entry is int
                    if isinstance(infl_list, int):
                        ids.add(int(infl_list))
                        continue

                    # Case C: dict influence
                    if isinstance(infl_list, dict):
                        for k in ("id", "bone", "bone_id", "group", "matrix"):
                            if k in infl_list:
                                try:
                                    ids.add(int(infl_list[k]))
                                except Exception:
                                    pass
                                break
                        continue

                    # Case D: tuple like (id, weight)
                    if isinstance(infl_list, tuple):
                        if len(infl_list) >= 1:
                            try:
                                ids.add(int(infl_list[0]))
                            except Exception:
                                pass
                        continue

                    # Case E: list/tuple of influences
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
                    else:
                        # unknown scalar type, ignore
                        pass

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

        # Absolute timeline cursor (used for geoset alpha + for deriving t_rel)
        t_abs = int(self.t_ms)

        # Sequence timing should come from ImportedModel (_load_from_db set these)
        seq_start = int(self._seq_start_ms)
        seq_dur = int(self._seq_dur_ms)

        # Evaluate pose: evaluator should sample bone channels with t_rel = t_abs - seq_start
        pose = self._evaluator.evaluate_pose(t_abs, seq_start, seq_dur)

        # Geoset visibility: absolute-time alpha + user checkbox
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

            self.gl.snapshot_default_camera()
            self._cam_init_done = True

        # Single pose submit (don’t call twice)
        self.gl.set_pose(pose, self._rig, active_ids=None)

        t_rel = int(t_abs - seq_start)
        pct = 0.0 if seq_dur <= 0 else (max(0.0, min(1.0, t_rel / float(seq_dur))) * 100.0)
        # Prefer the actual active sequence name if available; otherwise use sequence_name
        active_seq_name = getattr(getattr(self, "_evaluator", None), "sequence_name", None)
        if not active_seq_name:
            active_seq_name = getattr(self, "sequence_name", "(seq)")

        self.time_lbl.config(text=f"{active_seq_name}  {t_rel}ms  ({pct:.1f}%)")

        # Keep timeline slider in sync during playback (but never fight the user's drag)
        if not self._scrubbing:
            self._ignore_timeline_callback = True
            try:
                self.timeline_var.set(float(t_rel))
            finally:
                self._ignore_timeline_callback = False

        if self.debug_anim_var.get():
            print(f"[anim] seq={active_seq_name!r} dur={seq_dur} t_rel={t_rel} t_abs={t_abs}")
