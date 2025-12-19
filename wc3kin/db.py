from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UnitRow:
    id: int
    race: str
    unit_name: str
    unit_dir: str
    primary_model_path: str
    last_scanned: str


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )

    # schema v1
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS units (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          race TEXT NOT NULL,
          unit_name TEXT NOT NULL,
          unit_dir TEXT NOT NULL,
          primary_model_path TEXT NOT NULL,
          last_scanned TEXT NOT NULL,
          UNIQUE(race, unit_dir)
        );
        """
    )

    # placeholder for Step 2+
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS animations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          unit_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'original',
          created_at TEXT NOT NULL,
          UNIQUE(unit_id, name),
          FOREIGN KEY(unit_id) REFERENCES units(id) ON DELETE CASCADE
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bones (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          unit_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          UNIQUE(unit_id, name),
          FOREIGN KEY(unit_id) REFERENCES units(id) ON DELETE CASCADE
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bone_motion (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          animation_id INTEGER NOT NULL,
          bone_id INTEGER NOT NULL,
          t_min REAL, t_max REAL,
          r_min REAL, r_max REAL,
          motion_rms REAL,
          FOREIGN KEY(animation_id) REFERENCES animations(id) ON DELETE CASCADE,
          FOREIGN KEY(bone_id) REFERENCES bones(id) ON DELETE CASCADE,
          UNIQUE(animation_id, bone_id)
        );
        """
    )

    # version tracking
    con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?);", (str(SCHEMA_VERSION),))
    con.commit()


def upsert_units(con: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    """
    rows: (race, unit_name, unit_dir, primary_model_path, last_scanned_iso)
    Returns number of rows inserted/updated (approx).
    """
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO units (race, unit_name, unit_dir, primary_model_path, last_scanned)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(race, unit_dir) DO UPDATE SET
          unit_name=excluded.unit_name,
          primary_model_path=excluded.primary_model_path,
          last_scanned=excluded.last_scanned
        ;
        """,
        list(rows),
    )
    con.commit()
    return cur.rowcount


def get_races(con: sqlite3.Connection) -> list[str]:
    cur = con.execute("SELECT DISTINCT race FROM units ORDER BY race;")
    return [r["race"] for r in cur.fetchall()]


def get_units_for_race(con: sqlite3.Connection, race: str) -> list[tuple[int, str]]:
    """
    Returns list of (unit_id, unit_name)
    """
    cur = con.execute(
        """
        SELECT id, unit_name
        FROM units
        WHERE race = ?
        ORDER BY unit_name;
        """,
        (race,),
    )
    return [(int(r["id"]), str(r["unit_name"])) for r in cur.fetchall()]


def get_animations_for_unit(con: sqlite3.Connection, unit_id: int) -> list[str]:
    cur = con.execute(
        """
        SELECT name
        FROM animations
        WHERE unit_id = ?
        ORDER BY name;
        """,
        (unit_id,),
    )
    return [str(r["name"]) for r in cur.fetchall()]


def get_unit_detail(con: sqlite3.Connection, unit_id: int) -> Optional[UnitRow]:
    cur = con.execute(
        """
        SELECT id, race, unit_name, unit_dir, primary_model_path, last_scanned
        FROM units WHERE id = ?;
        """,
        (unit_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return UnitRow(
        id=int(row["id"]),
        race=str(row["race"]),
        unit_name=str(row["unit_name"]),
        unit_dir=str(row["unit_dir"]),
        primary_model_path=str(row["primary_model_path"]),
        last_scanned=str(row["last_scanned"]),
    )
