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
    warnings: List[str]
    error: Optional[str]
    # debugging
    blender_returncode: int
    blender_stdout: str
    blender_stderr: str
    input_used: str  # path actually extracted (may be cached FBX)


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
    """
    If model is .mdl/.mdx, convert it to a cached FBX using wc3_export_with_meshes.py (addon-powered).
    Returns the path to the FBX that should be used for extraction.
    """
    ext = model_abspath.suffix.lower()
    if ext not in (".mdl", ".mdx"):
        return model_abspath

    if wc3_json_root is None or not Path(wc3_json_root).exists():
        raise RuntimeError(
            "wc3_json_root is missing/invalid in config.json. "
            "It is required to convert .mdl/.mdx to FBX."
        )

    out_fbx = _cache_fbx_path(project_root, model_abspath)
    if out_fbx.exists() and out_fbx.stat().st_size > 0:
        return out_fbx

    converter_script = _project_script(project_root, "blender_scripts/wc3_export_with_meshes.py")
    if not converter_script.exists():
        raise FileNotFoundError(f"Missing converter script: {converter_script}")

    # The converter script expects --mdx, but it works as the input model path for both .mdl/.mdx.
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
        str(Path(wc3_json_root)),
        "--out-fbx",
        str(out_fbx),
    ]
    if texture_root and Path(texture_root).exists():
        cmd += ["--texture-root", str(Path(texture_root))]

    if logger:
        logger.info("Running WC3 convert (.mdl/.mdx -> .fbx): %s", " ".join(cmd))

    p = subprocess.run(cmd, capture_output=True, text=True)

    if logger:
        if p.stdout.strip():
            logger.info("WC3 convert stdout:\n%s", p.stdout.strip())
        if p.stderr.strip():
            logger.warning("WC3 convert stderr:\n%s", p.stderr.strip())

    if not out_fbx.exists() or out_fbx.stat().st_size == 0:
        raise RuntimeError(
            "WC3 conversion did not produce an FBX.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stdout:\n{p.stdout}\n\n"
            f"stderr:\n{p.stderr}"
        )

    return out_fbx


def pick_extractor_script(project_root: Path) -> Path:
    # After conversion, extraction always uses the generic extractor.
    return _project_script(project_root, "blender_scripts/kin_extract.py")


def run_blender_extract(
    blender_path: Path,
    project_root: Path,
    model_abspath: Path,
    wc3_json_root: Optional[Path] = None,
    texture_root: Optional[Path] = None,
    logger=None,
) -> ExtractResult:
    model_abspath = model_abspath.resolve()

    if not blender_path.exists():
        raise FileNotFoundError(f"Blender not found: {blender_path}")
    if not model_abspath.exists():
        raise FileNotFoundError(f"Model not found: {model_abspath}")

    # Convert WC3 models first (mdl/mdx -> fbx cache), then extract from the FBX
    model_for_extract = ensure_fbx_for_wc3_model(
        blender_path=blender_path,
        project_root=project_root,
        model_abspath=model_abspath,
        wc3_json_root=wc3_json_root,
        texture_root=texture_root,
        logger=logger,
    )

    blender_script = pick_extractor_script(project_root)
    if not blender_script.exists():
        raise FileNotFoundError(f"Blender extractor script not found: {blender_script}")

    with tempfile.TemporaryDirectory(prefix="wc3kin_") as td:
        out_json = Path(td) / "extract.json"

        cmd = [
            str(blender_path),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--input",
            str(model_for_extract),
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
                f"Command: {' '.join(cmd)}\n\n"
                f"stdout:\n{p.stdout}\n\n"
                f"stderr:\n{p.stderr}"
            )

        data = json.loads(out_json.read_text(encoding="utf-8"))
        return ExtractResult(
            ok=bool(data.get("ok")),
            armature_name=data.get("armature_name"),
            bones=list(data.get("bones") or []),
            animations=list(data.get("animations") or []),
            warnings=list(data.get("warnings") or []),
            error=data.get("error"),
            blender_returncode=int(p.returncode),
            blender_stdout=p.stdout or "",
            blender_stderr=p.stderr or "",
            input_used=str(model_for_extract),
        )


def ingest_extract_to_db(con: sqlite3.Connection, unit_id: int, extracted: ExtractResult) -> None:
    """
    Writes bones + animations for the unit. Uses INSERT OR IGNORE to keep idempotent.
    """
    now = _iso_now()
    cur = con.cursor()

    if extracted.bones:
        cur.executemany(
            """
            INSERT OR IGNORE INTO bones (unit_id, name)
            VALUES (?, ?);
            """,
            [(unit_id, b) for b in extracted.bones],
        )

    if extracted.animations:
        cur.executemany(
            """
            INSERT OR IGNORE INTO animations (unit_id, name, source, created_at)
            VALUES (?, ?, 'original', ?);
            """,
            [(unit_id, a, now) for a in extracted.animations],
        )

    con.commit()
