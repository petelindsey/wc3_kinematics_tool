
#wc3kin/viewer/mdl_mesh_parse.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from .mesh_provider import MeshData

_VEC2_RE = re.compile(r"\{\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\}")
_VEC3_RE = re.compile(r"\{\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\}")
_INT_RE = re.compile(r"-?\d+")

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def _parse_vec2_list(block: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for (a, b) in _VEC2_RE.findall(block)]

def _extract_block(text: str, name: str) -> list[str]:
    """
    Matches both:
        name { ... }
        name <n> { ... }
        name <n> <m> { ... }
        name 1 1140 { ... }   # (common in WC3 MDL Faces)
    Also tolerates commas in header counts.
    """
    import re

    blocks: list[str] = []
    i = 0

    pat = re.compile(rf"\b{re.escape(name)}\b(?:\s+[-\d,]+)*\s*\{{", re.MULTILINE)

    while True:
        m = pat.search(text, i)
        if not m:
            break

        k = text.find("{", m.start())
        if k < 0:
            break

        depth = 0
        for end in range(k, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[k + 1 : end])
                    i = end + 1
                    break
        else:
            break

    return blocks

    

def _parse_vec3_list(block: str) -> list[tuple[float, float, float]]:
    return [(float(a), float(b), float(c)) for (a, b, c) in _VEC3_RE.findall(block)]

def _parse_ints(block: str) -> list[int]:
    return [int(x) for x in _INT_RE.findall(block)]

def parse_mdl_mesh(mdl_path: Path) -> MeshData:
    text = _read_text(mdl_path)

    # Most WC3 models: mesh is in Geoset blocks.
    geosets = _extract_block(text, "Geoset")
    if not geosets:
        raise RuntimeError(f"No Geoset blocks found in {mdl_path}")

    # For now: take the first geoset. You can extend to multiple later.
    g = geosets[0]

    # Vertices
    v_blocks = _extract_block(g, "Vertices")
    if not v_blocks:
        print("[mdl_parse] Geoset head:", g[:200].replace("\n", "\\n"))
        raise RuntimeError("Geoset has no Vertices block")
    vertices = _parse_vec3_list(v_blocks[0])

    # Faces / Triangles
    faces_blocks = _extract_block(g, "Faces")
    if not faces_blocks:
        raise RuntimeError("Geoset has no Faces block")
    faces = faces_blocks[0]
    tri_blocks = _extract_block(faces, "Triangles")
    if not tri_blocks:
        raise RuntimeError("Faces block missing Triangles")
    tri_ints = _parse_ints(tri_blocks[0])
    if len(tri_ints) % 3 != 0:
        raise RuntimeError("Triangle index list is not a multiple of 3")
    triangles = [(tri_ints[i], tri_ints[i+1], tri_ints[i+2]) for i in range(0, len(tri_ints), 3)]

    # VertexGroup (optional but needed for skinning)
    vertex_groups = None
    vg_blocks = _extract_block(g, "VertexGroup")
    if vg_blocks:
        vertex_groups = _parse_ints(vg_blocks[0])
        # Some files may include count header; if mismatch, just keep what we got.

    # TVertices / UVs (optional)
    uvs = None
    tv_blocks = _extract_block(g, "TVertices")
    if tv_blocks:
        uvs = _parse_vec2_list(tv_blocks[0])

        # WC3 uses (u,v) with v typically top-down; OpenGL expects bottom-up.
        # You can flip here OR in the renderer. I recommend flipping in renderer
        # while prototyping; keep raw data intact for now.
        # Example flip would be: uvs = [(u, 1.0 - v) for (u, v) in uvs]

        # Basic sanity check: most models have same count
        if len(uvs) != len(vertices):
            print(f"[mdl_parse] WARNING: UV count ({len(uvs)}) != vertex count ({len(vertices)})")

    # Groups / Matrices (optional)
    groups_matrices = None
    groups_blocks = _extract_block(g, "Groups")
    if groups_blocks:
        gb = groups_blocks[0]
        matrices_blocks = _extract_block(gb, "Matrices")
        if matrices_blocks:
            groups_matrices = []
            for mb in matrices_blocks:
                groups_matrices.append(_parse_ints(mb))

    return MeshData(
        vertices=vertices,
        triangles=triangles,
        vertex_groups=vertex_groups,
        groups_matrices=groups_matrices,
        uvs=uvs,
    )
