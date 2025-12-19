#scanner.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FoundUnit:
    race: str
    unit_name: str
    unit_dir: Path
    primary_model_path: Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pick_primary_model(files: List[Path], ext_priority: List[str]) -> Optional[Path]:
    if not files:
        return None

    # group by ext priority
    ext_priority_l = [e.lower() for e in ext_priority]
    files_by_ext = {}
    for f in files:
        files_by_ext.setdefault(f.suffix.lower(), []).append(f)

    for ext in ext_priority_l:
        cands = files_by_ext.get(ext)
        if cands:
            # stable: shortest name then lexicographic
            cands_sorted = sorted(cands, key=lambda p: (len(p.name), p.name.lower()))
            return cands_sorted[0]

    # fallback: shortest name overall
    return sorted(files, key=lambda p: (len(p.name), p.name.lower()))[0]


def scan_units(units_root: Path, ext_priority: List[str]) -> Tuple[List[FoundUnit], List[str]]:
    warnings: List[str] = []

    if not units_root.exists():
        return [], [f"Units root not found: {units_root}"]

    if not units_root.is_dir():
        return [], [f"Units root is not a folder: {units_root}"]

    races = [p for p in units_root.iterdir() if p.is_dir()]
    if not races:
        return [], [f"No race folders found under: {units_root}"]

    found: List[FoundUnit] = []
    ext_set = {e.lower() for e in ext_priority}

    for race_dir in sorted(races, key=lambda p: p.name.lower()):
        race = race_dir.name

        # walk all subdirs; treat any dir containing model files as a unit folder
        for d in race_dir.rglob("*"):
            if not d.is_dir():
                continue

            model_files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ext_set]
            primary = _pick_primary_model(model_files, ext_priority)
            if not primary:
                continue

            # unit_name is the path under race folder
            try:
                unit_rel = d.relative_to(race_dir)
            except ValueError:
                warnings.append(f"Could not relativize path: {d} under {race_dir}")
                continue

            unit_name = str(unit_rel).replace("/", "\\")  # normalize for Windows display
            found.append(
                FoundUnit(
                    race=race,
                    unit_name=unit_name,
                    unit_dir=d,
                    primary_model_path=primary,
                )
            )

    # de-dup by (race, unit_dir)
    uniq = {}
    for u in found:
        key = (u.race, str(u.unit_dir).lower())
        # if duplicates, prefer one with "better" primary (earlier in ext list)
        if key not in uniq:
            uniq[key] = u
        else:
            # keep the one whose primary extension is higher priority
            def rank(p: Path) -> int:
                try:
                    return ext_priority.index(p.suffix.lower())
                except ValueError:
                    return 10_000

            if rank(u.primary_model_path) < rank(uniq[key].primary_model_path):
                uniq[key] = u

    return list(uniq.values()), warnings


def to_db_rows(found_units: Iterable[FoundUnit], units_root: Path) -> List[tuple]:
    now = _iso_now()
    rows: List[tuple] = []
    for u in found_units:
        # store unit_dir as relative to units_root for portability
        try:
            rel_dir = u.unit_dir.relative_to(units_root)
            rel_model = u.primary_model_path.relative_to(units_root)
        except ValueError:
            # if outside root, store absolute (still workable)
            rel_dir = u.unit_dir
            rel_model = u.primary_model_path

        rows.append(
            (
                u.race,
                u.unit_name,
                str(rel_dir),
                str(rel_model),
                now,
            )
        )
    return rows
