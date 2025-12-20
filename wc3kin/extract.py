from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ExtractResult:
    ok: bool
    armature_name: Optional[str]
    bones: List[str]
    animations: List[str]
    animations_source: str  # 'original' or 'per_action'
    warnings: List[str]
    error: Optional[str]
    blender_returncode: int
    blender_stdout: str
    blender_stderr: str
    input_used: str


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_script(project_root: Path, rel: str) -> Path:
    return (project_root / rel).resolve()


def _cache_fbx_path(project_root: Path, model_abspath: Path) -> Path:
    cache_dir = _project_script(project_root, "_cache_fbx")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (cache_dir / (model_abspath.stem + ".fbx")).resolve()


def ensure_fbx_for_wc3_model(
    blender_path: Path,
    project_root: Path,
    model_abspath: Path,
    wc3_json_root: Optional[Path],
    texture_root: Optional[Path],
    logger=None,
) -> Path:
    ext = model_abspath.suffix.lower()
    if ext not in (".mdl", ".mdx"):
        return model_abspath

    out_fbx = _cache_fbx_path(project_root, model_abspath)
    if out_fbx.exists() and out_fbx.stat().st_size > 0:
        return out_fbx

    converter_script = _project_script(project_root, "blender_scripts/wc3_export_with_meshes.py")

    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(converter_script),
        "--",
        "--mdx",
        str(model_abspath),
        "--model-name",
        model_abspath.stem,
        "--json-root",
        str(wc3_json_root),
        "--out-fbx",
        str(out_fbx),
    ]

    if texture_root:
        cmd += ["--texture-root", str(texture_root)]

    if logger:
        logger.info("Running WC3->FBX export: %s", " ".join(cmd))

    p = subprocess.run(cmd, capture_output=True, text=True)

    if logger:
        if p.stdout.strip():
            logger.info("WC3 export stdout:\n%s", p.stdout.strip())
        if p.stderr.strip():
            logger.warning("WC3 export stderr:\n%s", p.stderr.strip())

    if not out_fbx.exists():
        raise RuntimeError(f"WC3 conversion failed:\n{p.stderr}")

    return out_fbx


def pick_extractor_script(project_root: Path) -> Path:
    return _project_script(project_root, "blender_scripts/kin_extract.py")


def _run_extract_once(
    blender_path: Path,
    blender_script: Path,
    input_fbx: Path,
    logger=None,
) -> Tuple[dict, subprocess.CompletedProcess[str]]:
    """Run the Blender extractor script on a single FBX and return (json, process)."""
    with tempfile.TemporaryDirectory(prefix="wc3kin_") as td:
        out_json = Path(td) / "extract.json"

        cmd = [
            str(blender_path),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--input",
            str(input_fbx),
            "--output",
            str(out_json),
        ]

        if logger:
            logger.info("Running Blender extract: %s", " ".join(cmd))

        p = subprocess.run(cmd, capture_output=True, text=True)

        if logger:
            if p.stdout.strip():
                logger.info("Blender stdout:\n%s", p.stdout.strip())
            if p.stderr.strip():
                logger.warning("Blender stderr:\n%s", p.stderr.strip())

        if not out_json.exists():
            raise RuntimeError(
                "Extractor did not produce output JSON.\n"
                f"Input: {input_fbx}\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"stdout:\n{p.stdout}\n\n"
                f"stderr:\n{p.stderr}"
            )

        return json.loads(out_json.read_text(encoding="utf-8")), p


def _list_fbx_files(dir_path: Path) -> List[Path]:
    """List FBX files case-insensitively (handles .fbx / .FBX)."""
    if not dir_path.is_dir():
        return []
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".fbx"],
        key=lambda p: p.name.lower(),
    )


def _derive_anim_name_from_filename(unit_stem: str, anim_fbx: Path) -> str:
    """Derive <Action> from <Unit>_<Action>.fbx, else fallback to full stem."""
    stem = anim_fbx.stem
    prefix = unit_stem + "_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return stem


def run_blender_extract(
    blender_path: Path,
    project_root: Path,
    model_abspath: Path,
    wc3_json_root: Optional[Path] = None,
    texture_root: Optional[Path] = None,
    logger=None,
) -> ExtractResult:
    model_for_extract = ensure_fbx_for_wc3_model(
        blender_path,
        project_root,
        model_abspath,
        wc3_json_root,
        texture_root,
        logger,
    )

    blender_script = pick_extractor_script(project_root)

    # 1) Extract bones from base FBX
    base_fbx = Path(model_for_extract)
    base_data, base_proc = _run_extract_once(blender_path, blender_script, base_fbx, logger=logger)

    bones = list(base_data.get("bones") or [])
    armature_name = base_data.get("armature_name")
    warnings = list(base_data.get("warnings") or [])
    error = base_data.get("error")

    # 2) Extract animations from per-action FBXs, if present.
    #
    # Primary expected location:
    #   <base_fbx.parent>/<base_fbx.stem>_anims/*.fbx
    # Fallback (handles cases where base stem differs from original model stem):
    #   <project_root>/_cache_fbx/<model_abspath.stem>_anims/*.fbx
    expected_anim_dir = base_fbx.parent / f"{base_fbx.stem}_anims"
    fallback_anim_dir = _project_script(project_root, "_cache_fbx") / f"{model_abspath.stem}_anims"

    anim_dir = expected_anim_dir if expected_anim_dir.is_dir() else fallback_anim_dir

    animations: List[str] = []
    animations_source = "original"

    if logger:
        logger.info("Base FBX: %s", base_fbx)
        logger.info("Expected anim dir: %s (exists=%s)", expected_anim_dir, expected_anim_dir.is_dir())
        logger.info("Fallback anim dir: %s (exists=%s)", fallback_anim_dir, fallback_anim_dir.is_dir())
        logger.info("Using anim dir: %s (exists=%s)", anim_dir, anim_dir.is_dir())

    per_action_fbxs = _list_fbx_files(anim_dir)

    if per_action_fbxs:
        animations_source = "per_action"
        unit_stem = base_fbx.stem

        if logger:
            logger.info("Per-action FBXs found: %d", len(per_action_fbxs))
            for f in per_action_fbxs:
                logger.info("  anim fbx: %s", f)

        for fbx in per_action_fbxs:
            try:
                clip_data, _ = _run_extract_once(blender_path, blender_script, fbx, logger=logger)
                clip_anims = list(clip_data.get("animations") or [])

                if clip_anims:
                    animations.extend(clip_anims)
                else:
                    # Deterministic fallback if extractor didn't register an Action
                    animations.append(_derive_anim_name_from_filename(unit_stem, fbx))

                for w in list(clip_data.get("warnings") or []):
                    warnings.append(f"{fbx.name}: {w}")

                if clip_data.get("error"):
                    warnings.append(f"{fbx.name}: {clip_data.get('error')}")
            except Exception as e:
                warnings.append(f"{fbx.name}: {e!r}")
    else:
        # Legacy: animations embedded in the base FBX.
        # Note: this can include helper actions like Range_Nodes; per-action mode avoids that.
        animations = list(base_data.get("animations") or [])

        if logger:
            logger.info("No per-action FBXs found; using base FBX animations (%d).", len(animations))

    animations = sorted(set(animations), key=str.lower)

    return ExtractResult(
        ok=bool(base_data.get("ok")),
        armature_name=armature_name,
        bones=bones,
        animations=animations,
        animations_source=animations_source,
        warnings=warnings,
        error=error,
        blender_returncode=int(base_proc.returncode),
        blender_stdout=base_proc.stdout or "",
        blender_stderr=base_proc.stderr or "",
        input_used=str(model_for_extract),
    )


def ingest_extract_to_db(con: sqlite3.Connection, unit_id: int, extracted: ExtractResult) -> None:
    now = _iso_now()
    cur = con.cursor()

    if extracted.bones:
        cur.executemany(
            "INSERT OR IGNORE INTO bones (unit_id, name) VALUES (?, ?);",
            [(unit_id, b) for b in extracted.bones],
        )

    if extracted.animations:
        cur.executemany(
            """
            INSERT OR IGNORE INTO animations (unit_id, name, source, created_at)
            VALUES (?, ?, ?, ?);
            """,
            [(unit_id, a, extracted.animations_source or "original", now) for a in extracted.animations],
        )

    con.commit()
