# wc3kin/mdl_import.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sqlite3

@dataclass(frozen=True)
class MdlImportResult:
    ok: bool
    nodes_upserted: int = 0
    sequences_upserted: int = 0
    node_channels_upserted: int = 0
    geoset_anims_upserted: int = 0
    warnings: list[str] = None

def import_mdl_to_db(
    con: sqlite3.Connection,
    unit_id: int,
    model_abspath: Path,
    *,
    force_update: bool = False,
    logger=None,
) -> MdlImportResult:
    """
    Step-0/1 bridge:
    - If force_update=False and we already imported this mdl sha -> no-op.
    - Otherwise parse MDL and upsert mdl_* tables + (optionally) harvested_json kinds.
    Stub OK initially: just return ok=True.
    """
    # TODO: implement sha1 tracking, parsing, and table upserts
    return MdlImportResult(ok=True, warnings=[])