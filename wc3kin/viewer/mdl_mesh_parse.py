
#wc3kin/viewer/mdl_mesh_parse.py
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Iterable
from .mesh_provider import MeshData
import json
from pathlib import Path
from typing import Any


#_VEC2_RE = re.compile(r"\{\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\}")
_VEC3_RE = re.compile(r"\{\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\}")
_FM_RE = re.compile(r"\bFilterMode\s+([A-Za-z]+)", re.IGNORECASE)
_ALPHA_RE = re.compile(r"\bstatic\s+Alpha\s+([-0-9.eE]+)", re.IGNORECASE)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_INT_RE = re.compile(r"-?\d+")
_MATID_RE = re.compile(r"\bMaterialID\s+(\d+)", re.IGNORECASE)
_IMAGE_RE = re.compile(r'\bImage\s+"([^"]+)"', re.IGNORECASE)
_REPL_RE = re.compile(r"\bReplaceableId\s+(\d+)", re.IGNORECASE)
_TEXID_RE = re.compile(r"\bTextureID\s+(\d+)", re.IGNORECASE)
_LAYER_ALPHA_RE = re.compile(r"\b(?:static\s+)?Alpha\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
_FILTER_RE = re.compile(r"\bFilterMode\s+([A-Za-z]+)", re.IGNORECASE)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def _parse_ints(block: str, label: str):
    idx = re.search(rf"(?i)\b{re.escape(label)}\b", block)
    if not idx:
        return []
    start = block.find("{", idx.end())
    if start == -1:
        return []
    inner = _slice_brace(block, start)
    return [int(x) for x in _INT_RE.findall(inner)]

def _parse_vec2_list(block: str, label: str):
    idx = re.search(rf"(?i)\b{re.escape(label)}\b", block)
    if not idx:
        return []
    start = block.find("{", idx.end())
    if start == -1:
        return []
    inner = _slice_brace(block, start)
    out = []
    for m in re.finditer(r"\{([^{}]+)\}", inner):
        nums = _FLOAT_RE.findall(m.group(1))
        if len(nums) >= 2:
            out.append((float(nums[0]), float(nums[1])))
    return out

def _parse_vec3_list(block: str, label: str):
    idx = re.search(rf"(?i)\b{re.escape(label)}\b", block)
    if not idx:
        return []
    start = block.find("{", idx.end())
    if start == -1:
        return []
    inner = _slice_brace(block, start)
    out = []
    for m in re.finditer(r"\{([^{}]+)\}", inner):
        nums = _FLOAT_RE.findall(m.group(1))
        if len(nums) >= 3:
            out.append((float(nums[0]), float(nums[1]), float(nums[2])))
    return out

def _slice_brace(text: str, brace_idx: int) -> str:
    if brace_idx < 0 or brace_idx >= len(text) or text[brace_idx] != "{":
        return ""
    depth = 0
    i = brace_idx
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx + 1 : i]
        i += 1
    return ""

def _extract_block(text: str, name: str):
    """
    Matches both:
        name { ... }
        name <n> { ... }
        name <n> <m> { ... }
        name 1 1140 { ... }   # (common in WC3 MDL Faces)
    Also tolerates commas in header counts.
    """

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

def _extract_blocks(text: str, keyword: str):
    blocks = []
    pat = re.compile(rf"(?i)\b{re.escape(keyword)}\b")
    for m in pat.finditer(text):
        i = m.start()
        j = text.find("{", m.end())
        if j == -1:
            continue
        depth = 0
        k = j
        while k < len(text):
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[i : k + 1])
                    break
            k += 1
    return blocks


def _parse_groups_matrices(geoset_block: str):
    vertex_groups = _parse_ints(geoset_block, "VertexGroup")
    groups_matrices = []
    for mb in _extract_blocks(geoset_block, "Matrices"):
        brace = mb.find("{")
        if brace == -1:
            continue
        inner = _slice_brace(mb, brace)
        mats = [int(x) for x in _INT_RE.findall(inner)]
        if mats:
            groups_matrices.append(mats)
    return vertex_groups, groups_matrices


                            
def _parse_faces_to_tris(geoset_block: str):

    # WC3 MDL typically nests indices under:
    #   Faces ... { Triangles ... { {i,j,k}, ... } }
    m_faces = re.search(r"(?i)\bFaces\b", geoset_block)
    if not m_faces:
        return []
    brace_faces = geoset_block.find("{", m_faces.end())
    if brace_faces == -1:
        return []
    faces_inner = _slice_brace(geoset_block, brace_faces)

    m_tri = re.search(r"(?i)\bTriangles\b", faces_inner)
    if not m_tri:
        return []
    brace_tri = faces_inner.find("{", m_tri.end())
    if brace_tri == -1:
        return []
    tri_inner = _slice_brace(faces_inner, brace_tri)

    ints = [int(x) for x in _INT_RE.findall(tri_inner)]
    tris = []
    for i in range(0, len(ints) - 2, 3):
        tris.append((ints[i], ints[i + 1], ints[i + 2]))
    return tris


def _parse_textures_table(text: str):
    textures_block = _extract_blocks(text, "Textures")
    if not textures_block:
        return []
    tb = textures_block[0]
    out = []
    for bmp in _extract_blocks(tb, "Bitmap"):
        entry = {}
        mrep = _REPL_RE.search(bmp)
        if mrep:
            entry["replaceable_id"] = int(mrep.group(1))
        mim = _IMAGE_RE.search(bmp)
        if mim:
            entry["image"] = mim.group(1)
        out.append(entry)
    return out


def _parse_materials_table(text: str):
    mats_block = _extract_blocks(text, "Materials")
    if not mats_block:
        return []
    mb = mats_block[0]
    materials = []
    for mat in _extract_blocks(mb, "Material"):
        layers  = []
        for layer in _extract_blocks(mat, "Layer"):
            ld = {}
            mt = _TEXID_RE.search(layer)
            if mt:
                ld["texture_id"] = int(mt.group(1))
            mf = _FILTER_RE.search(layer)
            if mf:
                ld["filter_mode"] = mf.group(1)
            ma = _LAYER_ALPHA_RE.search(layer)
            if ma:
                try:
                    ld["alpha"] = float(ma.group(1))
                except Exception:
                    ld["alpha"] = 1.0
            else:
                ld["alpha"] = 1.0
            ld["two_sided"] = bool(re.search(r"(?i)\bTwoSided\b", layer))
            ld["unshaded"] = bool(re.search(r"(?i)\bUnshaded\b", layer))
            layers.append(ld)
        materials.append({"layers": layers})
    return materials


def extract_geoset_texture_name(mdl_text: str, geoset_index: int = 0):
    """
    Returns the Image string (basename) for the given geoset via:
      Geoset -> MaterialID -> Material -> TextureID -> Textures Bitmap Image

    Output example: "Arachnathid.blp" (keep .blp; your loader converts to .png later)
    """
    # 1) Textures list: Bitmap { Image "..." }
    tex_blocks = _extract_block(mdl_text, "Textures")
    if not tex_blocks:
        return None
    images = _IMAGE_RE.findall(tex_blocks[0])
    if not images:
        return None

    # 2) Materials in order (MaterialID indexes into this order)
    material_blocks = _extract_block(mdl_text, "Material")
    if not material_blocks:
        return None

    # Grab first TextureID used by each material (usually base layer)
    material_to_texid = []
    for mb in material_blocks:
        m = _TEXID_RE.search(mb)
        material_to_texid.append(int(m.group(1)) if m else None)

    # 3) Pick geoset, read its MaterialID
    geosets = _extract_block(mdl_text, "Geoset")
    if not geosets or geoset_index < 0 or geoset_index >= len(geosets):
        return None

    gm = _MATID_RE.search(geosets[geoset_index])
    if not gm:
        return None
    mat_id = int(gm.group(1))
    if mat_id < 0 or mat_id >= len(material_to_texid):
        return None

    tex_id = material_to_texid[mat_id]
    if tex_id is None or tex_id < 0 or tex_id >= len(images):
        return None

    # basename only; ignore directories (your rule)
    return os.path.basename(images[tex_id])


def export_mesh_truth_json(
    out_path: Path,
    *,
    geoset_index: int,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    extra: dict[str, Any] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idxs = [i for tri in triangles for i in tri]
    max_i = max(idxs) if idxs else -1
    min_i = min(idxs) if idxs else -1

    payload = {
        "geoset_index": geoset_index,
        "counts": {
            "verts": len(vertices),
            "tris": len(triangles),
            "uvs": len(uvs),
        },
        "index_stats": {
            "min": int(min_i),
            "max": int(max_i),
            "in_range": bool(max_i <= len(vertices) - 1 and min_i >= 0),
        },
        "bbox": {
            "min": [
                min(v[0] for v in vertices) if vertices else 0.0,
                min(v[1] for v in vertices) if vertices else 0.0,
                min(v[2] for v in vertices) if vertices else 0.0,
            ],
            "max": [
                max(v[0] for v in vertices) if vertices else 0.0,
                max(v[1] for v in vertices) if vertices else 0.0,
                max(v[2] for v in vertices) if vertices else 0.0,
            ],
        },
        "vertices": vertices,
        "triangles": triangles,
        "uvs": uvs,
        "extra": extra or {},
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def parse_mdl_mesh_all(mdl_path):
    mdl_path = Path(mdl_path)
    text = _read_text(mdl_path)
    geosets = _extract_blocks(text, "Geoset")
    if not geosets:
        raise RuntimeError(f"No Geoset blocks found in {mdl_path}")

    textures_table = _parse_textures_table(text)
    materials_table = _parse_materials_table(text)

    out = []
    for gi, g in enumerate(geosets):
        mm = _MATID_RE.search(g)
        geoset_material_id = int(mm.group(1)) if mm else None

        vertices = _parse_vec3_list(g, "Vertices")
        uvs = _parse_vec2_list(g, "TVertices")

        vertex_groups, groups_matrices = _parse_groups_matrices(g)
        triangles = _parse_faces_to_tris(g)
        if gi == 3:
            export_mesh_truth_json(
                Path("debug_exports") / "viewer_parse_geoset3.json",
                geoset_index=gi,
                vertices=vertices,
                triangles=triangles,
                uvs=uvs,
                extra={"source": str(mdl_path)},
            )

        texture_name = extract_geoset_texture_name(text, geoset_index=gi)

        out.append(
            MeshData(
                vertices=vertices,
                triangles=triangles,
                vertex_groups=vertex_groups,
                groups_matrices=groups_matrices,
                uvs=uvs,
                geoset_material_id=geoset_material_id,
                textures=textures_table,
                texture_name=texture_name,
                materials=materials_table,
            )
        )

    return out

def parse_mdl_mesh(mdl_path: Path) -> MeshData:
    return parse_mdl_mesh_all(mdl_path)[0]
