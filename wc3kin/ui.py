from __future__ import annotations

import logging
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from pathlib import Path
import subprocess
import json

from .config import load_config
from .db import (
    connect,
    init_db,
    get_races,
    get_units_for_race,
    get_animations_for_unit,
    get_unit_detail,
    upsert_units,
)
from .extract import run_blender_extract, ingest_extract_to_db
from .scanner import scan_units, to_db_rows


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
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.title("WC3 Units Kinematics Library")
        self.geometry("980x620")

        self.cfg = load_config(config_path)
        self.logger = _setup_logger(self.cfg.log_path)

        self.con = connect(self.cfg.db_path)
        init_db(self.con)

        # UI state
        self.race_var = tk.StringVar(value="")
        self.unit_var = tk.StringVar(value="")
        self.anim_var = tk.StringVar(value="")

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

        ttk.Label(row, text="Animation").pack(side="left")
        self.anim_cb = ttk.Combobox(row, textvariable=self.anim_var, state="readonly", width=28)
        self.anim_cb.pack(side="left", padx=(6, 0))

        # buttons
        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Import / Rescan Units", command=self._import_rescan).pack(side="left")
        ttk.Button(btns, text="Ingest Animations (Selected)", command=self._ingest_selected).pack(side="left", padx=(10, 0))
        ttk.Button(
            btns,
            text="Ingest All (This Race)",
            command=lambda: self._ingest_all(only_selected_race=True),
        ).pack(side="left", padx=4)
        
        ttk.Button(
            btns,
            text="Ingest Animations (All Units)",
            command=lambda: self._ingest_all(only_selected_race=False),
        ).pack(side="left", padx=(10, 0))

        # status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0, 6))

        # info panel
        info_frame = ttk.LabelFrame(self, text="Output / Errors (copy-paste friendly)")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_text = tk.Text(info_frame, wrap="word")
        self.info_text.pack(fill="both", expand=True, padx=8, pady=8)

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

        # list[(unit_id, unit_name)]
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

        # --- fetch unit detail ---
        detail = get_unit_detail(self.con, uid)
        if not detail:
            self._write_info("Failed to load unit detail from DB.")
            return

        # --- fetch animations ---
        anims = get_animations_for_unit(self.con, uid)
        self.anim_cb["values"] = anims
        if anims:
            self.anim_var.set(anims[0])

        # --- print info ---
        self._write_info(
            f"Selected Unit:\n"
            f"  Race: {detail.race}\n"
            f"  Unit: {detail.unit_name}\n"
            f"  Unit Dir: {detail.unit_dir}\n"
            f"  Primary Model: {detail.primary_model_path}\n"
            f"  Last Scanned: {detail.last_scanned}\n\n"
            f"Animations in DB: {len(anims)}\n"
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

    def _ingest_unit_id(self, uid: int) -> bool:
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
                print(msg)
                self._write_info(msg, append=True)
                self._set_status("Ingest failed.")
                return False

            ingest_extract_to_db(self.con, uid, extracted)
            self._write_info(
                f"Ingest OK: {detail.unit_name}\n"
                f"  Used: {extracted.input_used}\n"
                f"  Bones: {len(extracted.bones)}\n"
                f"  Animations: {len(extracted.animations)}\n",
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
            print(msg)
            self._write_info(msg, append=True)
            self._set_status("Ingest exception.")
            return False

        finally:
            # refresh animation dropdown for currently selected unit
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

def run_app() -> None:
    here = Path(__file__).resolve().parent.parent
    config_path = here / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at: {config_path}")

    app = App(config_path=config_path)
    app.mainloop()
