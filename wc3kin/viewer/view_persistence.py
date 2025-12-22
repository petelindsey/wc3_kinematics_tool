# wc3kin/viewer/view_persistence.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sqlite3


def _db_file_from_connection(con: sqlite3.Connection) -> Optional[Path]:
    try:
        row = con.execute("PRAGMA database_list;").fetchone()
        # row = (seq, name, file)
        if row and len(row) >= 3 and row[2]:
            return Path(row[2])
    except Exception:
        return None
    return None


def default_persistence_path(con: sqlite3.Connection) -> Path:
    db_path = _db_file_from_connection(con)
    if db_path is None:
        # fallback: cwd
        return Path("wc3kin_viewer_persistence.json").resolve()
    return db_path.with_suffix(db_path.suffix + ".viewer_persistence.json")


@dataclass
class ViewerPersist:
    path: Path
    data: Dict[str, Any]

    @staticmethod
    def load(path: Path) -> "ViewerPersist":
        if path.exists():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return ViewerPersist(path=path, data=obj)
            except Exception:
                pass
        return ViewerPersist(path=path, data={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @staticmethod
    def key(unit_id: int, sequence_name: str) -> str:
        return f"{int(unit_id)}::{sequence_name}"

    def get_camera(self, unit_id: int, sequence_name: str) -> Optional[Dict[str, Any]]:
        k = self.key(unit_id, sequence_name)
        v = self.data.get(k)
        return v if isinstance(v, dict) else None

    def set_camera(self, unit_id: int, sequence_name: str, cam_state: Dict[str, Any]) -> None:
        k = self.key(unit_id, sequence_name)
        self.data[k] = cam_state
