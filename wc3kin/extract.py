from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import sqlite3


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

    p = subprocess.run(cmd, capture_output=True, text=True)

    if not out_fbx.exists():
        raise RuntimeError(f"WC3 conversion failed:\n{p.stderr}")

    return out_fbx


def pick_extractor_script(project_root: Path) -> Path:
    return _project_script(project_root, "blender_scripts/kin_extract.py")


def _run_extract_once(
    blender_path: Path,
    blender_script: Path,
    input_fbx: Path,
) -> tuple[dict, subprocess.CompletedProcess[str]]:
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

        p = subprocess.run(cmd, capture_output=True, text=True)

        if not out_json.exists():
            raise RuntimeError(f"Extractor failed:\n{p.stderr}")

        return json.loads(out_json.read_text()), p



def _find_per_action_anim_dir(model_for_extract: Path) -> Path | None:
    """
    Find the per-action animation directory for a cached FBX.

    Expected: <cache_dir>/<Unit>_anims/
    But on Windows / manual copying, casing can differ, so we match case-insensitively.
    Also supports a heuristic fallback: any *_anims dir that contains FBXs starting with '<Unit>_'.
    """
    cache_dir = model_for_extract.parent
    unit_stem = model_for_extract.stem
    expected_name = f"{unit_stem}_anims"

    # 1) Exact expected path
    expected = cache_dir / expected_name
    if expected.is_dir():
        return expected

    # 2) Case-insensitive directory name match in cache_dir
    for p in cache_dir.iterdir():
        if p.is_dir() and p.name.lower() == expected_name.lower():
            return p

    # 3) Heuristic: pick the *_anims dir with the most FBXs that look like '<Unit>_<Clip>.fbx'
    best_dir: Path | None = None
    best_count = 0
    for p in cache_dir.iterdir():
        if not p.is_dir() or not p.name.lower().endswith("_anims"):
            continue
        fbxs = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".fbx"]
        count = sum(1 for f in fbxs if f.stem.lower().startswith(unit_stem.lower() + "_"))
        if count > best_count:
            best_count = count
            best_dir = p

    return best_dir
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
    base_data, base_proc = _run_extract_once(
        blender_path, blender_script, Path(model_for_extract)
    )

    bones = list(base_data.get("bones") or [])
    armature_name = base_data.get("armature_name")
    warnings = list(base_data.get("warnings") or [])
    error = base_data.get("error")

    # 2) Extract animations from per-action FBXs
    # 2) Extract animations from per-action FBXs
    model_for_extract_path = Path(model_for_extract)
    anim_dir = _find_per_action_anim_dir(model_for_extract_path)
    animations: list[str] = []
    animations_source = "original"

    if anim_dir and anim_dir.is_dir():
        animations_source = "per_action"
        # enumerate FBXs case-insensitively (handles .FBX)
        fbxs = sorted([p for p in anim_dir.iterdir() if p.is_file() and p.suffix.lower() == ".fbx"], key=lambda p: p.name.lower())
        for fbx in fbxs:
            try:
                clip_data, _ = _run_extract_once(blender_path, blender_script, fbx)
                clip_anims = clip_data.get("animations") or []
                if clip_anims:
                    animations.extend(clip_anims)
                else:
                    animations.append(fbx.stem.replace(model_for_extract_path.stem + "_", ""))
            except Exception as e:
                warnings.append(f"{fbx.name}: {e!r}")

    else:
        # Legacy: animations embedded in the base FBX (may include helper actions like Range_Nodes).
        animations = list(base_data.get("animations") or [])
        if anim_dir is None:
            warnings.append(f"No per-action anim directory found for {model_for_extract_path.name}; using base FBX animations.")
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
            [(unit_id, a, extracted.animations_source, now) for a in extracted.animations],
        )

    con.commit()
