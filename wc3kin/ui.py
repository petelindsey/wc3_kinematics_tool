from __future__ import annotations

import logging
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk
import sqlite3
import json


from .viewer.window import ViewerWindow
from .config import load_config
from .db import (
    connect,
    init_db,
    get_races,
    get_units_for_race,
    get_sequences_for_unit,
    get_unit_detail,
    upsert_units,
    ingest_sequences_from_harvest_json,
    ingest_known_harvested_json_blobs,
    HarvestJsonError,
)

from .extract import run_blender_extract, ingest_extract_to_db
from .scanner import scan_units, to_db_rows
from .kinematics import compute_motion_stats_for_unit


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wc3kin")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class App(tk.Tk):
    BASE_GEOM = "980x620"
    SQL_GEOM = "980x820"

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.title("WC3 Units Kinematics Library")
        self.geometry(self.BASE_GEOM)

        self.cfg = load_config(config_path)
        self.logger = _setup_logger(self.cfg.log_path)

        self.con = connect(self.cfg.db_path)
        init_db(self.con)

        # UI state
        self.race_var = tk.StringVar(value="")
        self.unit_var = tk.StringVar(value="")
        self.anim_var = tk.StringVar(value="")
        self.show_death_var = tk.BooleanVar(value=False)
        self.json_only_var = tk.BooleanVar(value=True)
        self.force_update_var = tk.BooleanVar(value=False) 

        # menu
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        ingest_menu = tk.Menu(menubar, tearoff=0)
        ingest_menu.add_checkbutton(
            label="Force Update",
            variable=self.force_update_var,
            onvalue=True,
            offvalue=False,
        )
        ingest_menu.add_checkbutton(
            label="JSON Only",
            variable=self.json_only_var,
            onvalue=True,
            offvalue=False,
        )
        ingest_menu.add_separator()
        ingest_menu.add_command(label="Ingest Sequences (Selected)", command=self._menu_ingest_sequences_selected)
        ingest_menu.add_command(
            label="Ingest Sequences (This Race)",
            command=lambda: self._menu_ingest_sequences_all(only_selected_race=True),
        )
        ingest_menu.add_command(
            label="Ingest Sequences (All Units)",
            command=lambda: self._menu_ingest_sequences_all(only_selected_race=False),
        )
        ingest_menu.add_separator()
        ingest_menu.add_command(label="Compute Motion Stats (Selected)", command=self._menu_motion_selected)
        ingest_menu.add_command(
            label="Compute Motion Stats (This Race)",
            command=lambda: self._menu_motion_all(only_selected_race=True),
        )
        ingest_menu.add_command(
            label="Compute Motion Stats (All Units)",
            command=lambda: self._menu_motion_all(only_selected_race=False),
        )
        ingest_menu.add_separator()
        menubar.add_cascade(label="Ingest", menu=ingest_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="SQL Console", command=self._toggle_sql_console)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Viewer", command=self._open_viewer)


        self.unit_id_by_name: dict[str, int] = {}

        self._build_ui()
        self._refresh_races()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        # dropdown row
        row = ttk.Frame(top)
        row.pack(fill="x")

        ttk.Label(row, text="Race").pack(side="left")
        self.race_cb = ttk.Combobox(row, textvariable=self.race_var, state="readonly", width=18)
        self.race_cb.pack(side="left", padx=(6, 16))
        self.race_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_race_changed())

        ttk.Label(row, text="Unit").pack(side="left")
        self.unit_cb = ttk.Combobox(row, textvariable=self.unit_var, state="readonly", width=42)
        self.unit_cb.pack(side="left", padx=(6, 16))
        self.unit_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_unit_changed())

        ttk.Label(row, text="Sequence").pack(side="left")
        self.anim_cb = ttk.Combobox(row, textvariable=self.anim_var, state="readonly", width=28)
        self.anim_cb.pack(side="left", padx=(6, 8))

        self.show_death_cb = ttk.Checkbutton(
            row,
            text="Show death/corpse",
            variable=self.show_death_var,
            command=self._on_unit_changed,
        )
        self.show_death_cb.pack(side="left")
        self.force_update_cb = ttk.Checkbutton(
            row,
            text="Force update",
            variable=self.force_update_var,
        )
        self.force_update_cb.pack(side="left", padx=(10, 0))
        # buttons
        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Import / Rescan Units", command=self._import_rescan).pack(side="left")
        ttk.Button(btns, text="Ingest (Selected)", command=self._ingest_selected).pack(side="left", padx=(10, 0))
        ttk.Button(
            btns,
            text="Ingest All (This Race)",
            command=lambda: self._ingest_all(only_selected_race=True),
        ).pack(side="left", padx=4)


        ttk.Button(
            btns,
            text="Ingest (All Units)",
            command=lambda: self._ingest_all(only_selected_race=False),
        ).pack(side="left", padx=(10, 0))

        ttk.Button(
            btns,
            text="SQL Console",
            command=self._toggle_sql_console,
        ).pack(side="right")

        # status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0, 6))

        # info panel
        info_frame = ttk.LabelFrame(self, text="Output / Errors (copy-paste friendly)")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_text = tk.Text(info_frame, wrap="word")
        self.info_text.pack(fill="both", expand=True, padx=8, pady=8)

        # SQL console (hidden by default)
        self.sql_frame = ttk.LabelFrame(self, text="SQL Console (use carefully)")
        self.sql_visible = False

        self.sql_query = tk.Text(self.sql_frame, height=5, wrap="word")
        self.sql_query.pack(fill="x", expand=False, padx=8, pady=(8, 4))

        sql_btn_row = ttk.Frame(self.sql_frame)
        sql_btn_row.pack(fill="x", padx=8, pady=(0, 4))

        ttk.Button(sql_btn_row, text="Run", command=self._run_sql).pack(side="left")
        ttk.Button(sql_btn_row, text="Clear", command=self._clear_sql_console).pack(side="left", padx=(8, 0))
        ttk.Button(sql_btn_row, text="Copy JSON", command=self._copy_sql_output_json).pack(side="left", padx=(8, 0))
        ttk.Button(sql_btn_row, text="Close", command=self._toggle_sql_console).pack(side="left", padx=(8, 0))

        self.sql_output = tk.Text(self.sql_frame, height=8, wrap="word")
        self.sql_output.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.sql_last_headers: list[str] = []
        self.sql_last_rows: list[dict[str, object]] = []

    def _open_viewer(self) -> None:
        unit_name = self.unit_var.get().strip()
        uid = self.unit_id_by_name.get(unit_name)
        if not uid:
            self._write_info("Viewer: no unit selected.", append=True)
            return

        seq_name = self.anim_var.get().strip()
        if not seq_name:
            self._write_info("Viewer: no sequence selected.", append=True)
            return

        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info("Viewer: failed to load unit detail from DB.", append=True)
            return

        model_abspath = (self.cfg.units_root / detail.primary_model_path).resolve()

        # Best-effort: ensure harvested blobs exist in DB (bones/boneanims at minimum)
        try:
            ingest_known_harvested_json_blobs(self.con, uid, model_abspath)
        except Exception as e:
            self._write_info(f"Viewer: blob ingest warning (continuing): {e!r}", append=True)

        try:
            ViewerWindow(
                self,
                con=self.con,
                units_root=self.cfg.units_root,
                unit_id=uid,
                sequence_name=seq_name,
            )
        except Exception as e:
            self.logger.exception("Viewer open failed")
            self._write_info(f"Viewer failed:\n{e!r}\n{traceback.format_exc()}", append=True)

    def _copy_sql_output_json(self) -> None:
        import json

        if not self.sql_last_headers:
            # No captured result-set; fall back to copying the raw text output
            txt = self.sql_output.get("1.0", "end").strip()
            if not txt:
                return
            self.clipboard_clear()
            self.clipboard_append(json.dumps({"text": txt}, ensure_ascii=False, indent=2))
            return

        payload = {
            "columns": self.sql_last_headers,
            "rows": self.sql_last_rows,
            "row_count": len(self.sql_last_rows),
        }
        self.clipboard_clear()
        self.clipboard_append(json.dumps(payload, ensure_ascii=False, indent=2))

    def _split_sql_statements(self,script: str) -> list[str]:
        """
        Split SQL script into complete statements using sqlite3.complete_statement.
        Handles semicolons inside strings better than naive split(';').
        """
        import sqlite3

        stmts: list[str] = []
        buf: list[str] = []
        for line in script.splitlines():
            buf.append(line)
            joined = "\n".join(buf).strip()
            if joined and sqlite3.complete_statement(joined):
                stmts.append(joined)
                buf = []
        tail = "\n".join(buf).strip()
        if tail:
            stmts.append(tail)
        return [s.strip() for s in stmts if s.strip()]

    def _clear_sql_console(self) -> None:
        self.sql_query.delete("1.0", "end")
        self.sql_output.delete("1.0", "end")
        self.sql_last_headers = []
        self.sql_last_rows = []

    def _set_status(self, s: str) -> None:
        self.status_var.set(s)
        self.update_idletasks()

    def _write_info(self, s: str, append: bool = True) -> None:
        if not append:
            self.info_text.delete("1.0", "end")
        self.info_text.insert("end", s.rstrip() + "\n")
        self.info_text.see("end")
        self.update_idletasks()

    def _refresh_races(self) -> None:
        races = get_races(self.con)
        self.race_cb["values"] = races
        if races:
            self.race_var.set(races[0])
            self._on_race_changed()

    def _on_race_changed(self) -> None:
        race = self.race_var.get().strip()
        self.unit_var.set("")
        self.anim_var.set("")
        self.anim_cb["values"] = []

        if not race:
            self.unit_cb["values"] = []
            return

        units = get_units_for_race(self.con, race)
        self.unit_id_by_name = {name: uid for uid, name in units}
        unit_names = [name for _, name in units]
        self.unit_cb["values"] = unit_names

        if unit_names:
            self.unit_var.set(unit_names[0])
            self._on_unit_changed()
        else:
            self._write_info(f"No units stored for race: {race}")

    def _on_unit_changed(self) -> None:
        unit_name = self.unit_var.get().strip()
        uid = self.unit_id_by_name.get(unit_name)

        self.anim_var.set("")
        self.anim_cb["values"] = []

        if not uid:
            self._write_info("No unit selected.")
            return

        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info("Failed to load unit detail from DB.")
            return

        seqs = get_sequences_for_unit(self.con, uid, include_death_and_corpse=bool(self.show_death_var.get()))
        self.anim_cb["values"] = seqs
        if seqs:
            self.anim_var.set(seqs[0])

        self._write_info(
            f"Selected Unit:\n"
            f"  Race: {detail.race}\n"
            f"  Unit: {detail.unit_name}\n"
            f"  Unit Dir: {detail.unit_dir}\n"
            f"  Primary Model: {detail.primary_model_path}\n"
            f"  Last Scanned: {detail.last_scanned}\n\n"
            f"Sequences in DB: {len(seqs)}\n"
        )

    def _import_rescan(self) -> None:
        try:
            self._set_status("Scanning Units folder...")
            units = scan_units(self.cfg.units_root, self.cfg.model_extensions_priority)
            rows = to_db_rows(units)
            upsert_units(self.con, rows)
            self._set_status(f"Import complete. Units found: {len(rows)}")
            self._refresh_races()
        except Exception as e:
            self.logger.exception("Import/rescan failed")
            self._write_info("Import/rescan failed:\n" + repr(e) + "\n" + traceback.format_exc(), append=True)
            self._set_status("Import/rescan failed.")

    def _ingest_selected(self) -> None:
        unit_name = self.unit_var.get().strip()
        uid = self.unit_id_by_name.get(unit_name)
        if not uid:
            self._write_info("Ingest failed: no unit selected.", append=True)
            self._set_status("Ingest failed.")
            return

        self._ingest_unit_id(uid)

    def _copy_sql_output_json(self) -> None:
        import json

        if not self.sql_last_headers:
            # No captured result-set; fall back to copying the raw text output
            txt = self.sql_output.get("1.0", "end").strip()
            if not txt:
                return
            self.clipboard_clear()
            self.clipboard_append(json.dumps({"text": txt}, ensure_ascii=False, indent=2))
            return

        payload = {
            "columns": self.sql_last_headers,
            "rows": self.sql_last_rows,
            "row_count": len(self.sql_last_rows),
        }
        self.clipboard_clear()
        self.clipboard_append(json.dumps(payload, ensure_ascii=False, indent=2))

    def _menu_motion_selected(self) -> None:
        unit_name = self.unit_var.get().strip()
        uid = self.unit_id_by_name.get(unit_name)
        if not uid:
            self._write_info("Motion stats failed: no unit selected.", append=True)
            self._set_status("Motion stats failed.")
            return
        self._motion_unit_id(uid)

    def _menu_motion_all(self, *, only_selected_race: bool) -> None:
        if only_selected_race and self.race_var.get().strip():
            race = self.race_var.get().strip()
            cur = self.con.execute("SELECT id FROM units WHERE race = ? ORDER BY race, unit_name;", (race,))
        else:
            cur = self.con.execute("SELECT id FROM units ORDER BY race, unit_name;")

        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            self._write_info("No units in DB. Run Import / Rescan Units first.", append=True)
            return

        self._write_info(f"Starting motion stats for {len(ids)} unit(s)...", append=True)
        ok = 0
        failed = 0
        for i, uid in enumerate(ids, start=1):
            self._set_status(f"Motion stats {i}/{len(ids)} ...")
            if self._motion_unit_id(uid):
                ok += 1
            else:
                failed += 1
        self._set_status(f"Motion stats done. OK={ok}, Failed={failed}")
        self._write_info(f"Motion stats done. OK={ok}, Failed={failed}", append=True)

    def _motion_unit_id(self, uid: int) -> bool:
        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info(f"Motion stats failed: missing unit detail for id={uid}", append=True)
            return False

        model_abspath = (self.cfg.units_root / detail.primary_model_path).resolve()

        try:
            self._set_status(f"Computing motion stats (JSON only): {detail.unit_name} ...")
            res = compute_motion_stats_for_unit(
                self.con,
                uid,
                model_abspath,
                include_death_and_corpse=bool(self.show_death_var.get()),
            )
            self._write_info(
                f"Motion stats OK: {detail.unit_name}\n"
                f"  Bones upserted: {res.bones_upserted}\n"
                f"  Bone motion rows upserted: {res.motion_rows_upserted}\n"
                f"  Sequence agg rows upserted: {res.seq_rows_upserted}\n",
                append=True,
            )
            self._set_status(f"Motion stats OK: {detail.unit_name}")
            return True
        except HarvestJsonError as e:
            self._write_info(
                f"Motion stats failed for {detail.unit_name}\n"
                f"Model: {model_abspath}\n"
                f"Error: {e}\n",
                append=True,
            )
            self._set_status("Motion stats failed.")
            return False
        except Exception as e:
            self.logger.exception("Motion stats exception")
            self._write_info(
                f"Motion stats exception for {detail.unit_name}\n"
                f"Model: {model_abspath}\n"
                f"{repr(e)}\n\n{traceback.format_exc()}",
                append=True,
            )
            self._set_status("Motion stats exception.")
            return False
        finally:
            self._on_unit_changed()

    def _menu_ingest_sequences_selected(self) -> None:
        """Ingest sequences for the selected unit.

        When "JSON Only" is enabled, this uses the harvested JSON fast-path and does not require Blender.
        When disabled, it falls back to the full ingest pipeline (which may invoke Blender).
        """
        if self.force_update_var.get():
            self._ingest_selected()
            return
        if self.json_only_var.get():
            self._ingest_sequences_json_selected()
        else:
            # Full ingest pipeline (legacy behavior)
            self._ingest_selected()

    def _menu_ingest_sequences_all(self, *, only_selected_race: bool) -> None:
        """Batch-ingest sequences for a race or all units, respecting the JSON-only toggle."""
        if self.force_update_var.get():
            self._ingest_all()
            return
        if self.json_only_var.get():
            self._ingest_sequences_json_all(only_selected_race=only_selected_race)
        else:
            self._ingest_all(only_selected_race=only_selected_race)

    def _ingest_sequences_json_selected(self) -> None:
        """Fast path: ingest canonical sequence ranges from harvested JSON only.

        This does not require Blender and only touches the `sequences` table.
        """
        unit_name = self.unit_var.get().strip()
        uid = self.unit_id_by_name.get(unit_name)
        if not uid:
            self._write_info("JSON-only sequence ingest failed: no unit selected.", append=True)
            self._set_status("JSON-only sequence ingest failed.")
            return

        self._ingest_sequences_json_unit_id(uid)

    def _ingest_sequences_json_unit_id(self, uid: int) -> bool:
        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info(f"JSON-only sequence ingest failed: missing unit detail for id={uid}", append=True)
            return False

        model_abspath = (self.cfg.units_root / detail.primary_model_path).resolve()
        try:
            self._set_status(f"Ingesting sequences (JSON only): {detail.unit_name} ...")
            upserts = ingest_sequences_from_harvest_json(self.con, uid, model_abspath)
            if upserts <= 0:
                self._write_info(
                    f"JSON-only sequence ingest: no sequences upserted for {detail.unit_name}\n"
                    f"  Expected: {model_abspath.with_name(model_abspath.stem + '_materialsets.json')}\n",
                    append=True,
                )
                self._set_status(f"JSON-only sequence ingest: no sequences for {detail.unit_name}")
                return False

            self._write_info(
                f"JSON-only sequence ingest OK: {detail.unit_name}\n"
                f"  Sequences upserted: {upserts}\n"
                f"  Source: {model_abspath.with_name(model_abspath.stem + '_materialsets.json')}\n",
                append=True,
            )
            self._set_status(f"JSON-only sequence ingest OK: {detail.unit_name}")
            return True
        except HarvestJsonError as e:
            self._write_info(
                f"JSON-only sequence ingest failed for {detail.unit_name}\n"
                f"Model: {model_abspath}\n"
                f"Error: {e}\n",
                append=True,
            )
            self._set_status("JSON-only sequence ingest failed.")
            return False
        except Exception as e:
            self.logger.exception("JSON-only sequence ingest exception")
            self._write_info(
                f"JSON-only sequence ingest exception for {detail.unit_name}\n"
                f"Model: {model_abspath}\n"
                f"{repr(e)}\n\n{traceback.format_exc()}",
                append=True,
            )
            self._set_status("JSON-only sequence ingest exception.")
            return False
        finally:
            self._on_unit_changed()

    def _ingest_unit_id(self, uid: int) -> bool:
        force_update = bool(self.force_update_var.get())
        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info(f"Ingest failed: missing unit detail for id={uid}", append=True)
            return False

        if not self.cfg.blender_path or not Path(self.cfg.blender_path).exists():
            self._write_info("Ingest failed: blender_path missing/invalid in config.json", append=True)
            return False

        model_abspath = (self.cfg.units_root / detail.primary_model_path).resolve()
        project_root = Path(__file__).resolve().parent.parent

        try:
            self._set_status(f"Ingesting: {detail.unit_name} ...")

            extracted = run_blender_extract(
                blender_path=Path(self.cfg.blender_path),
                project_root=project_root,
                model_abspath=model_abspath,
                wc3_json_root=self.cfg.wc3_json_root,
                texture_root=self.cfg.texture_root,
                logger=self.logger,
                force_update=force_update
            )

            if extracted.error:
                msg = (
                    f"Ingest failed for {detail.unit_name}\n"
                    f"Model: {model_abspath}\n"
                    f"Extractor input used: {extracted.input_used}\n"
                    f"Return code: {extracted.blender_returncode}\n"
                    f"Error: {extracted.error}\n"
                    f"Warnings: {extracted.warnings}\n\n"
                    f"--- Blender stdout ---\n{extracted.blender_stdout}\n\n"
                    f"--- Blender stderr ---\n{extracted.blender_stderr}\n"
                )
                self._write_info(msg, append=True)
                self._set_status("Ingest failed.")
                return False

            # Step 2 placeholder (bones + Blender-derived animations)
            ingest_extract_to_db(self.con, uid, extracted)

            # Canonical WC3 sequences from harvested JSONs (preferred)
            seq_upserts = ingest_sequences_from_harvest_json(self.con, uid, model_abspath)

            self._write_info(
                f"Ingest OK: {detail.unit_name}\n"
                f"  Used: {extracted.input_used}\n"
                f"  Bones: {len(extracted.bones)}\n"
                f"  Blender Animations (legacy): {len(extracted.animations)}\n"
                f"  Sequences upserted (harvest JSON): {seq_upserts}\n",
                append=True,
            )
            self._set_status(f"Ingest OK: {detail.unit_name}")
            return True

        except Exception as e:
            self.logger.exception("Ingest exception")
            msg = (
                f"Ingest exception for {detail.unit_name}\n"
                f"Model: {model_abspath}\n"
                f"{repr(e)}\n\n{traceback.format_exc()}"
            )
            self._write_info(msg, append=True)
            self._set_status("Ingest exception.")
            return False

        finally:
            self._on_unit_changed()

    def _ingest_all(self, only_selected_race: bool = False) -> None:
        if only_selected_race and self.race_var.get().strip():
            race = self.race_var.get().strip()
            if not race:
                self._write_info("Select a race first.", append=True)
                return
            cur = self.con.execute(
                "SELECT id FROM units WHERE race = ? ORDER BY race, unit_name;",
                (race,),
            )
        else:
            cur = self.con.execute("SELECT id FROM units ORDER BY race, unit_name;")

        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            self._write_info("No units in DB. Run Import / Rescan Units first.", append=True)
            return

        self._write_info(f"Starting batch ingest for {len(ids)} unit(s)...", append=True)

        ok = 0
        failed = 0
        for i, uid in enumerate(ids, start=1):
            self._set_status(f"Batch ingest {i}/{len(ids)} ...")
            if self._ingest_unit_id(uid):
                ok += 1
            else:
                failed += 1

        self._set_status(f"Batch ingest done. OK={ok}, Failed={failed}")
        self._write_info(f"Batch ingest done. OK={ok}, Failed={failed}", append=True)

    def _ingest_sequences_json_all(self, only_selected_race: bool = False) -> None:
        """Batch JSON-only sequence ingest."""
        if only_selected_race and self.race_var.get().strip():
            race = self.race_var.get().strip()
            if not race:
                self._write_info("Select a race first.", append=True)
                return
            cur = self.con.execute(
                "SELECT id FROM units WHERE race = ? ORDER BY race, unit_name;",
                (race,),
            )
        else:
            cur = self.con.execute("SELECT id FROM units ORDER BY race, unit_name;")

        ids = [int(r["id"]) for r in cur.fetchall()]
        if not ids:
            self._write_info("No units in DB. Run Import / Rescan Units first.", append=True)
            return

        self._write_info(f"Starting JSON-only sequence ingest for {len(ids)} unit(s)...", append=True)

        ok = 0
        failed = 0
        for i, uid in enumerate(ids, start=1):
            self._set_status(f"Sequence ingest (JSON only) {i}/{len(ids)} ...")
            if self._ingest_sequences_json_unit_id(uid):
                ok += 1
            else:
                failed += 1

        self._set_status(f"JSON-only sequence ingest done. OK={ok}, Failed={failed}")
        self._write_info(f"JSON-only sequence ingest done. OK={ok}, Failed={failed}", append=True)

    # ---------------- SQL console ----------------

    def _toggle_sql_console(self) -> None:
        if self.sql_visible:
            # hide
            self.sql_frame.pack_forget()
            self.sql_visible = False
            self.geometry(self.BASE_GEOM)
        else:
            # show (clear both boxes every time)
            self.sql_query.delete("1.0", "end")
            self.sql_output.delete("1.0", "end")
            self.sql_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))
            self.sql_visible = True
            self.geometry(self.SQL_GEOM)

    def _run_sql(self) -> None:


        script = self.sql_query.get("1.0", "end").strip()
        self.sql_output.delete("1.0", "end")
        self.sql_last_headers = []
        self.sql_last_rows = []

        if not script:
            self.sql_output.insert("end", "No SQL provided.\n")
            return

        try:
            stmts = self._split_sql_statements(script)

            total_selects = 0
            for i, sql in enumerate(stmts, start=1):
                first = sql.lstrip().split(None, 1)[0].lower() if sql.lstrip() else ""
                self.sql_output.insert("end", f"-- [{i}/{len(stmts)}] {first.upper()} --\n")

                if first in {"select", "pragma", "with"}:
                    cur = self.con.execute(sql)
                    rows = cur.fetchall()
                    headers = [d[0] for d in cur.description] if cur.description else []

                    # Print table
                    if headers:
                        self.sql_output.insert("end", "\t".join(headers) + "\n")
                        self.sql_output.insert("end", "-" * 80 + "\n")
                        for r in rows:
                            self.sql_output.insert("end", "\t".join(str(r[h]) for h in headers) + "\n")
                    self.sql_output.insert("end", f"Rows: {len(rows)}\n\n")

                    # Save the most recent SELECT result for Copy JSON
                    total_selects += 1
                    self.sql_last_headers = headers
                    self.sql_last_rows = [{h: r[h] for h in headers} for r in rows]

                else:
                    self.con.execute(sql)
                    self.con.commit()
                    self.sql_output.insert("end", "OK (executed and committed)\n\n")

            if total_selects == 0:
                self.sql_output.insert("end", "Done. (No result-set statements)\n")

        except sqlite3.Error as e:
            self.sql_output.insert("end", f"SQLite error: {e!r}\n")
        except Exception as e:
            self.sql_output.insert("end", f"Error: {e!r}\n")

def run_app() -> None:
    here = Path(__file__).resolve().parent.parent
    config_path = here / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at: {config_path}")

    app = App(config_path=config_path)
    app.mainloop()