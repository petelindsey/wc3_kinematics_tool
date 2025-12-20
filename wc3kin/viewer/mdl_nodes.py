# wc3kin/viewer/mdl_nodes.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

NODE_TYPES = [
    "Bone",
    "Helper",
    "Attachment",
    "Light",
    "EventObject",
    "ParticleEmitter",
    "ParticleEmitter2",
    "RibbonEmitter",
    "Camera",
    "CollisionShape",
]

NODE_HEADER_RE = re.compile(
    r'^\s*(' + "|".join(re.escape(t) for t in NODE_TYPES) + r')\s+"([^"]+)"\s*\{',
    re.I,
)
OBJECT_ID_RE = re.compile(r"\bObjectId\s+(-?\d+)", re.I)
PARENT_RE = re.compile(r"\bParent\s+(-?\d+)", re.I)
PIVOT_RE = re.compile(r"\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}")


@dataclass(frozen=True)
class MdlNode:
    object_id: int
    parent_id: Optional[int]
    name: str
    type: str
    pivot: Tuple[float, float, float]


def _parse_pivots(text: str) -> List[Tuple[float, float, float]]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith("PivotPoints"):
            block_lines = [lines[i]]
            depth = lines[i].count("{") - lines[i].count("}")
            i += 1
            while i < n and depth > 0:
                l = lines[i]
                depth += l.count("{") - l.count("}")
                block_lines.append(l)
                i += 1
            block_text = "\n".join(block_lines)
            pivots: List[Tuple[float, float, float]] = []
            for pm in PIVOT_RE.finditer(block_text):
                pivots.append((float(pm.group(1)), float(pm.group(2)), float(pm.group(3))))
            return pivots
        i += 1
    return []


def load_nodes_from_mdl(mdl_path: Path) -> Dict[int, MdlNode]:
    """
    Viewer-only: parse MDL to get the full node graph + pivots, keyed by object_id.
    """
    text = mdl_path.read_text(encoding="utf-8", errors="ignore")
    pivots = _parse_pivots(text)

    nodes: Dict[int, MdlNode] = {}

    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = NODE_HEADER_RE.match(line)
        if not m:
            i += 1
            continue

        node_type = m.group(1)
        name = m.group(2).strip()

        block_lines = [line]
        depth = line.count("{") - line.count("}")
        i += 1
        while i < n and depth > 0:
            l = lines[i]
            depth += l.count("{") - l.count("}")
            block_lines.append(l)
            i += 1
        block_text = "\n".join(block_lines)

        oid_m = OBJECT_ID_RE.search(block_text)
        if not oid_m:
            continue
        oid = int(oid_m.group(1))

        pid_m = PARENT_RE.search(block_text)
        pid = int(pid_m.group(1)) if pid_m else None

        pivot = (0.0, 0.0, 0.0)
        if 0 <= oid < len(pivots):
            pivot = pivots[oid]

        nodes[oid] = MdlNode(
            object_id=oid,
            parent_id=pid,
            name=name,
            type=node_type,
            pivot=pivot,
        )

    return nodes
