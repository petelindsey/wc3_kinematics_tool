from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from wc3kin.viewer.types import SequenceDef

SCHEMA_VERSION = 3


class HarvestJsonError(RuntimeError):
    """Raised when a harvested JSON file exists but cannot be parsed/used."""


@dataclass(frozen=True)
class UnitRow:
    id: int
    race: str
    unit_name: str
    unit_dir: str
    primary_model_path: str
    last_scanned: str


# --- NEW DB HELPERS (add near other helpers) ---
def get_harvested_json_blob(con: sqlite3.Connection, unit_id: int, kind: str) -> Optional[dict]:
    """
    Fetch a raw harvested JSON blob from the `harvested_json` table and parse it.
    Returns None if missing.
    """
    cur = con.execute(
        """
        SELECT json_text
        FROM harvested_json
        WHERE unit_id = ? AND kind = ?;
        """,
        (int(unit_id), str(kind)),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(str(row["json_text"]))
    except Exception as e:
        raise HarvestJsonError(f"Failed to parse harvested_json blob for unit_id={unit_id}, kind={kind}: {e!r}") from e


def get_sequence_detail(con: sqlite3.Connection, unit_id: int, name: str) -> Optional["SequenceDef"]:
    """
    Fetch a sequence row from DB and return a viewer-friendly SequenceDef.
    """
    from .viewer.types import SequenceDef  # local import to avoid circular import

    cur = con.execute(
        """
        SELECT name, start, end, category, is_death, is_corpse
        FROM sequences
        WHERE unit_id = ? AND name = ?;
        """,
        (int(unit_id), str(name)),
    )
    row = cur.fetchone()
    if not row:
        return None
    return SequenceDef(
        name=str(row["name"]),
        start_ms=int(row["start"]),
        end_ms=int(row["end"]),
        category=str(row["category"]),
        is_death=bool(int(row["is_death"])),
        is_corpse=bool(int(row["is_corpse"])),
    )

def get_bone_stats_for_sequence(con: sqlite3.Connection, unit_id: int, sequence_id: int) -> list[sqlite3.Row]:
    cur = con.execute(
        """
        SELECT
          bms.bone_object_id AS bone_object_id,
          COALESCE(b.name, '') AS bone_name,
          bms.trans_rms_vel,
          bms.rot_ang_range_rad,
          bms.rot_rms_ang_vel
        FROM bone_motion_stats bms
        LEFT JOIN bones b
          ON b.unit_id = bms.unit_id
         AND b.object_id = bms.bone_object_id
        WHERE bms.unit_id = ?
          AND bms.sequence_id = ?
        ;
        """,
        (unit_id, sequence_id),
    )
    return list(cur.fetchall())

def upsert_sequence_fingerprint(
    con: sqlite3.Connection,
    unit_id: int,
    sequence_id: int,
    fingerprint_json: str,
    fingerprint_sha1: str,
) -> None:
    con.execute(
        """
        INSERT INTO sequence_fingerprints (unit_id, sequence_id, fingerprint_json, fingerprint_sha1)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(unit_id, sequence_id) DO UPDATE SET
          fingerprint_json = excluded.fingerprint_json,
          fingerprint_sha1 = excluded.fingerprint_sha1
        ;
        """,
        (unit_id, sequence_id, fingerprint_json, fingerprint_sha1),
    )
    con.commit()

def get_harvested_json_text(con: sqlite3.Connection, unit_id: int, kind: str) -> Optional[str]:
    cur = con.execute(
        "SELECT json_text FROM harvested_json WHERE unit_id = ? AND kind = ?;",
        (unit_id, kind),
    )
    row = cur.fetchone()
    return str(row["json_text"]) if row else None


def get_sequences_rows_for_unit(
    con: sqlite3.Connection,
    unit_id: int,
    *,
    include_death_and_corpse: bool = False,
) -> list[sqlite3.Row]:
    if include_death_and_corpse:
        cur = con.execute(
            "SELECT id, name, start, end, is_death, is_corpse FROM sequences WHERE unit_id = ? ORDER BY name;",
            (unit_id,),
        )
    else:
        cur = con.execute(
            """
            SELECT id, name, start, end, is_death, is_corpse
            FROM sequences
            WHERE unit_id = ?
              AND is_death = 0
              AND is_corpse = 0
            ORDER BY name;
            """,
            (unit_id,),
        )
    return list(cur.fetchall())


def upsert_bones_by_object_id(con: sqlite3.Connection, rows: list[tuple]) -> int:
    """
    rows: (unit_id, name, object_id, parent_object_id, pivot_x, pivot_y, pivot_z, depth, path)
    Requires UNIQUE(unit_id, object_id) to exist (or a unique index).
    """
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO bones (unit_id, name, object_id, parent_object_id, pivot_x, pivot_y, pivot_z, depth, path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, object_id) DO UPDATE SET
          name=excluded.name,
          parent_object_id=excluded.parent_object_id,
          pivot_x=excluded.pivot_x,
          pivot_y=excluded.pivot_y,
          pivot_z=excluded.pivot_z,
          depth=excluded.depth,
          path=excluded.path
        ;
        """,
        rows,
    )
    con.commit()
    return cur.rowcount


def upsert_bone_motion_stats(con: sqlite3.Connection, rows: list[tuple]) -> int:
    """
    rows match bone_motion_stats columns exactly
    """
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO bone_motion_stats (
          unit_id, sequence_id, bone_object_id,
          tx_min, tx_max, ty_min, ty_max, tz_min, tz_max,
          tmag_min, tmag_max,
          trans_rms_vel,
          rot_ang_range_rad, rot_rms_ang_vel,
          sx_min, sx_max, sy_min, sy_max, sz_min, sz_max,
          slice_start_ms, slice_end_ms, sample_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, sequence_id, bone_object_id) DO UPDATE SET
          tx_min=excluded.tx_min, tx_max=excluded.tx_max,
          ty_min=excluded.ty_min, ty_max=excluded.ty_max,
          tz_min=excluded.tz_min, tz_max=excluded.tz_max,
          tmag_min=excluded.tmag_min, tmag_max=excluded.tmag_max,
          trans_rms_vel=excluded.trans_rms_vel,
          rot_ang_range_rad=excluded.rot_ang_range_rad,
          rot_rms_ang_vel=excluded.rot_rms_ang_vel,
          sx_min=excluded.sx_min, sx_max=excluded.sx_max,
          sy_min=excluded.sy_min, sy_max=excluded.sy_max,
          sz_min=excluded.sz_min, sz_max=excluded.sz_max,
          slice_start_ms=excluded.slice_start_ms,
          slice_end_ms=excluded.slice_end_ms,
          sample_count=excluded.sample_count
        ;
        """,
        rows,
    )
    con.commit()
    return cur.rowcount


def upsert_sequence_motion_stats(con: sqlite3.Connection, rows: list[tuple]) -> int:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT INTO sequence_motion_stats (
          unit_id, sequence_id,
          active_bone_count, total_trans_rms, total_rot_rms
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, sequence_id) DO UPDATE SET
          active_bone_count=excluded.active_bone_count,
          total_trans_rms=excluded.total_trans_rms,
          total_rot_rms=excluded.total_rot_rms
        ;
        """,
        rows,
    )
    con.commit()
    return cur.rowcount


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

    # schema v3: raw harvested JSON blobs (foundation for future sequence-aware kinematics)
    #
    # We intentionally do not parse these into normalized tables yet. The goal is to preserve
    # the canonical harvested artifacts alongside the unit so later pipeline stages can be
    # driven by (sequence_name, start, end) from `sequences` without re-harvesting.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS harvested_json (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          unit_id INTEGER NOT NULL,
          kind TEXT NOT NULL,
          source_path TEXT NOT NULL,
          json_text TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(unit_id, kind),
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

    try:
        data = json.loads(mat_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HarvestJsonError(f"Failed to parse harvested JSON: {mat_path} ({e!r})") from e
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


def ingest_known_harvested_json_blobs(
    con: sqlite3.Connection,
    unit_id: int,
    model_abspath: Path,
    *,
    include_missing: bool = False,
) -> dict[str, int]:
    """Best-effort ingest of raw harvested JSON files.

    This is a *foundation hook* for future sequence-aware kinematics extraction.
    It is safe to run at any time: it only touches the `harvested_json` table.

    Returns a dict of kind -> rows affected (0/1).
    """

    stem = model_abspath.stem
    candidates = {
        "materialsets": model_abspath.with_name(f"{stem}_materialsets.json"),
        "bones": model_abspath.with_name(f"{stem}_bones.json"),
        "boneanims": model_abspath.with_name(f"{stem}_boneanims.json"),
        "animslices": model_abspath.with_name(f"{stem}_animslices.json"),
        "geosets": model_abspath.with_name(f"{stem}_geosets.json"),
        "meshsets": model_abspath.with_name(f"{stem}_meshsets.json"),
    }

    out: dict[str, int] = {}
    for kind, p in candidates.items():
        if not p.is_file():
            if include_missing:
                out[kind] = 0
            continue
        try:
            txt = p.read_text(encoding="utf-8")
            # Validate that it is JSON so downstream stages can assume it.
            json.loads(txt)
        except Exception as e:
            raise HarvestJsonError(f"Failed to parse harvested JSON: {p} ({e!r})") from e
        out[kind] = upsert_harvested_json_blob(con, unit_id, kind=kind, source_path=p, json_text=txt)
    return out


def upsert_harvested_json_blob(
    con: sqlite3.Connection,
    unit_id: int,
    *,
    kind: str,
    source_path: Path,
    json_text: str,
) -> int:
    """Store raw harvested JSON for later sequence-aware pipeline stages.

    This is intentionally a "blob" store (no parsing/normalization yet).
    """
    now = datetime.now(timezone.utc).isoformat()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO harvested_json (unit_id, kind, source_path, json_text, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(unit_id, kind) DO UPDATE SET
          source_path=excluded.source_path,
          json_text=excluded.json_text,
          created_at=excluded.created_at
        ;
        """,
        (unit_id, str(kind), str(source_path), str(json_text), now),
    )
    con.commit()
    return cur.rowcount


def ingest_harvested_json_blobs_for_unit(
    con: sqlite3.Connection,
    unit_id: int,
    model_abspath: Path,
    *,
    kinds: Optional[list[str]] = None,
) -> int:
    """Ingest any additional harvested JSONs adjacent to the unit model.

    Supported naming convention: <stem>_<kind>.json
    Example kinds: bones, boneanims, animslices, geosets, meshsets, materialsets

    Note: This function is a future-proofing hook and is not required for the
    sequence-only UI ingest.
    """
    stem = model_abspath.stem
    if kinds is None:
        kinds = ["bones", "boneanims", "animslices", "geosets", "meshsets", "materialsets"]

    upserts = 0
    for kind in kinds:
        p = model_abspath.with_name(f"{stem}_{kind}.json")
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8")
            # Ensure it's valid JSON before storing so downstream stages can trust it.
            json.loads(txt)
        except Exception as e:
            raise HarvestJsonError(f"Failed to parse harvested JSON: {p} ({e!r})") from e
        upserts += upsert_harvested_json_blob(con, unit_id, kind=kind, source_path=p, json_text=txt)

    return upserts
