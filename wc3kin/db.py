from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


SCHEMA_VERSION = 2


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


def _get_schema_version(con: sqlite3.Connection) -> int:
    try:
        cur = con.execute("SELECT value FROM meta WHERE key='schema_version';")
        row = cur.fetchone()
        if not row:
            return 0
        return int(row["value"])
    except Exception:
        return 0


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

    # schema v2: canonical WC3 sequences from harvested JSON
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sequences (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          unit_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          start INTEGER NOT NULL,
          end INTEGER NOT NULL,
          category TEXT NOT NULL DEFAULT 'unknown',
          is_death INTEGER NOT NULL DEFAULT 0,
          is_corpse INTEGER NOT NULL DEFAULT 0,
          source TEXT NOT NULL DEFAULT 'harvest_json',
          created_at TEXT NOT NULL,
          UNIQUE(unit_id, name),
          FOREIGN KEY(unit_id) REFERENCES units(id) ON DELETE CASCADE
        );
        """
    )

    # version tracking + migration bump
    con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1');")
    current = _get_schema_version(con)
    if current < SCHEMA_VERSION:
        con.execute(
            "UPDATE meta SET value=? WHERE key='schema_version';",
            (str(SCHEMA_VERSION),),
        )
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
    # Legacy list (Blender-derived)
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


def get_sequences_for_unit(
    con: sqlite3.Connection,
    unit_id: int,
    include_death_and_corpse: bool = False,
) -> list[str]:
    if include_death_and_corpse:
        cur = con.execute(
            """
            SELECT name
            FROM sequences
            WHERE unit_id = ?
            ORDER BY name;
            """,
            (unit_id,),
        )
    else:
        cur = con.execute(
            """
            SELECT name
            FROM sequences
            WHERE unit_id = ?
              AND is_death = 0
              AND is_corpse = 0
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


def ingest_sequences_from_harvest_json(
    con: sqlite3.Connection,
    unit_id: int,
    model_abspath: Path,
) -> int:
    """
    Read harvested JSONs next to the model and upsert canonical WC3 sequences.

    We primarily use <stem>_materialsets.json because it contains:
      - sequences: name -> {start,end}
      - sequence_category: name -> category (death/corpse/...)
    """
    stem = model_abspath.stem
    mat_path = model_abspath.with_name(f"{stem}_materialsets.json")
    if not mat_path.is_file():
        # nothing to ingest
        return 0

    data = json.loads(mat_path.read_text(encoding="utf-8"))
    seq_ranges = data.get("sequences") or {}
    seq_cat = data.get("sequence_category") or {}

    now = datetime.now(timezone.utc).isoformat()

    rows: list[tuple] = []
    for name, rng in seq_ranges.items():
        try:
            start = int(rng.get("start"))
            end = int(rng.get("end"))
        except Exception:
            continue
        cat = str(seq_cat.get(name) or "unknown")
        is_death = 1 if cat.lower() == "death" else 0
        is_corpse = 1 if cat.lower() == "corpse" else 0
        rows.append((unit_id, str(name), start, end, cat, is_death, is_corpse, now))

    if not rows:
        return 0

    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO sequences (unit_id, name, start, end, category, is_death, is_corpse, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, name) DO UPDATE SET
          start=excluded.start,
          end=excluded.end,
          category=excluded.category,
          is_death=excluded.is_death,
          is_corpse=excluded.is_corpse
        ;
        """,
        rows,
    )
    con.commit()
    return cur.rowcount
