from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import Counter
from typing import Any


# Expand this list as needed — these are common WC3 node types that participate in Parent chains.
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


def parse_pivot_points(text: str) -> list[tuple[float, float, float]]:
    lines = text.splitlines()
    n = len(lines)
    pivots: list[tuple[float, float, float]] = []
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
            for pm in PIVOT_RE.finditer(block_text):
                pivots.append((float(pm.group(1)), float(pm.group(2)), float(pm.group(3))))
            return pivots
        i += 1

    return pivots


def parse_nodes(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    n = len(lines)
    i = 0
    out: list[dict[str, Any]] = []

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

        out.append(
            {
                "type": node_type,
                "name": name,
                "object_id": oid,
                "parent_id": pid,
            }
        )

    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python mdl_nodes_test.py <path-to-.mdl>")
        return 2

    mdl_path = Path(sys.argv[1]).expanduser().resolve()
    if not mdl_path.exists():
        print("File not found:", mdl_path)
        return 2

    text = mdl_path.read_text(encoding="utf-8", errors="ignore")
    pivots = parse_pivot_points(text)
    nodes = parse_nodes(text)

    print("=== MDL ===")
    print("path:", mdl_path)
    print("pivotpoints:", len(pivots))
    print("nodes parsed:", len(nodes))
    print("node types seen:", sorted({n["type"] for n in nodes}))

    # attach pivots
    for n in nodes:
        oid = n["object_id"]
        n["pivot"] = pivots[oid] if 0 <= oid < len(pivots) else None

    id_set = {n["object_id"] for n in nodes}

    # hierarchy stats
    with_parent = sum(1 for n in nodes if n["parent_id"] is not None)
    missing = Counter()
    drawable_edges = 0
    for n in nodes:
        pid = n["parent_id"]
        if pid is None:
            continue
        if pid in id_set:
            drawable_edges += 1
        else:
            missing[pid] += 1

    print("\n=== HIERARCHY ===")
    print("nodes with parent_id:", with_parent)
    print("missing parent refs (unique):", len(missing))
    if missing:
        print("top missing parent ids:", missing.most_common(15))
    print("drawable parent->child edges:", drawable_edges)

    # show a few “bad” nodes
    if missing:
        missing_ids = set(missing.keys())
        print("\n=== SAMPLE NODES WITH MISSING PARENT (first 10) ===")
        shown = 0
        for n in nodes:
            if n["parent_id"] in missing_ids:
                print({k: n[k] for k in ("type", "name", "object_id", "parent_id", "pivot")})
                shown += 1
                if shown >= 10:
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
