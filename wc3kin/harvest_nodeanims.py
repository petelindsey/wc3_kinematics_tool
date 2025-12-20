# wc3kin/harvest_nodeanims.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Reuse node typing conventions from viewer/mdl_nodes.py (same list)
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

# Track block headers inside a node
TRACK_HEAD_RE = re.compile(r"\b(Translation|Rotation|Scaling)\b\s+(\d+)\s*\{", re.I)

# Key line:  <time>: { ... }
KEY_RE = re.compile(r"(\d+)\s*:\s*\{([^}]*)\}", re.I)

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class NodeAnim:
    object_id: int
    name: str
    type: str
    parent_id: Optional[int]
    translation: list[dict[str, Any]]
    rotation: list[dict[str, Any]]   # uses "quat"
    scaling: list[dict[str, Any]]


def _extract_balanced_block(text: str, open_brace_index: int) -> tuple[str, int]:
    """
    Given an index pointing at a '{', return (block_text_including_braces, end_index_exclusive).
    """
    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        raise ValueError("open_brace_index must point at '{'")
    depth = 0
    i = open_brace_index
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index : i + 1], i + 1
        i += 1
    raise ValueError("Unbalanced braces while extracting block")


def _parse_track_keys(track_block_text: str, kind: str) -> list[dict[str, Any]]:
    """
    track_block_text includes outer braces.
    kind in {"translation","rotation","scaling"}.
    Returns list of key dicts (time_ms + value/quat).
    """
    out: list[dict[str, Any]] = []

    for m in KEY_RE.finditer(track_block_text):
        t = int(m.group(1))
        body = m.group(2)
        floats = [float(x) for x in FLOAT_RE.findall(body)]
        if kind in ("translation", "scaling"):
            if len(floats) < 3:
                continue
            out.append({"time_ms": t, "value": [floats[0], floats[1], floats[2]]})
        else:
            # rotation
            if len(floats) < 4:
                continue
            out.append({"time_ms": t, "quat": [floats[0], floats[1], floats[2], floats[3]]})

    out.sort(key=lambda k: int(k["time_ms"]))
    return out


def _extract_tracks_from_node_block(node_block_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (translation_keys, rotation_keys, scaling_keys).
    If multiple Translation/Rotation/Scaling blocks exist, concatenates in file order.
    """
    tkeys: list[dict[str, Any]] = []
    rkeys: list[dict[str, Any]] = []
    skeys: list[dict[str, Any]] = []

    # Scan for "Translation N {" etc, then balanced-extract its block.
    idx = 0
    while True:
        m = TRACK_HEAD_RE.search(node_block_text, idx)
        if not m:
            break
        head_kind = m.group(1).lower()
        # m.end()-1 should be '{' or we search for next '{'
        brace_i = node_block_text.find("{", m.end() - 1)
        if brace_i < 0:
            idx = m.end()
            continue

        block, end_i = _extract_balanced_block(node_block_text, brace_i)
        if head_kind == "translation":
            tkeys.extend(_parse_track_keys(block, "translation"))
        elif head_kind == "rotation":
            rkeys.extend(_parse_track_keys(block, "rotation"))
        elif head_kind == "scaling":
            skeys.extend(_parse_track_keys(block, "scaling"))

        idx = end_i

    # ensure deterministic ordering
    tkeys.sort(key=lambda k: int(k["time_ms"]))
    rkeys.sort(key=lambda k: int(k["time_ms"]))
    skeys.sort(key=lambda k: int(k["time_ms"]))
    return tkeys, rkeys, skeys


def harvest_nodeanims_from_mdl(mdl_path: Path) -> dict[str, Any]:
    """
    Parse MDL text directly and produce nodeanims payload.
    """
    text = mdl_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    n = len(lines)

    nodes: list[NodeAnim] = []
    i = 0
    while i < n:
        line = lines[i]
        mh = NODE_HEADER_RE.match(line)
        if not mh:
            i += 1
            continue

        node_type = mh.group(1)
        name = mh.group(2).strip()

        # capture node block by brace depth (same strategy as viewer/mdl_nodes.py :contentReference[oaicite:4]{index=4})
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
        object_id = int(oid_m.group(1))

        pid_m = PARENT_RE.search(block_text)
        parent_id = int(pid_m.group(1)) if pid_m else None

        tkeys, rkeys, skeys = _extract_tracks_from_node_block(block_text)

        nodes.append(
            NodeAnim(
                object_id=object_id,
                name=name,
                type=node_type,
                parent_id=parent_id,
                translation=tkeys,
                rotation=rkeys,
                scaling=skeys,
            )
        )

    # Convert to JSON-serializable dicts in a shape compatible with evaluator
    node_objs: list[dict[str, Any]] = []
    for na in sorted(nodes, key=lambda x: x.object_id):
        node_objs.append(
            {
                "object_id": na.object_id,
                "name": na.name,
                "type": na.type,
                "parent_id": na.parent_id,
                "channels": {
                    "translation": na.translation,
                    "rotation": na.rotation,
                    "scaling": na.scaling,
                },
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "kind": "nodeanims",
        "mdl": str(mdl_path),
        "generated_at": now,
        "nodes": node_objs,
        # Compatibility mirror: lets you reuse build_anims_from_boneanims_json immediately
        "bones": node_objs,
    }
    return payload


def _is_non_identity_key(key: dict[str, Any], kind: str, eps: float = 1e-6) -> bool:
    if kind in ("translation", "scaling"):
        v = key.get("value")
        if not isinstance(v, list) or len(v) < 3:
            return False
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        if kind == "translation":
            return (abs(x) > eps) or (abs(y) > eps) or (abs(z) > eps)
        # scaling identity = (1,1,1)
        return (abs(x - 1.0) > eps) or (abs(y - 1.0) > eps) or (abs(z - 1.0) > eps)

    q = key.get("quat")
    if not isinstance(q, list) or len(q) < 4:
        return False
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return (abs(x) > eps) or (abs(y) > eps) or (abs(z) > eps) or (abs(w - 1.0) > eps)


def print_nodeanim_counts(payload: dict[str, Any], *, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> None:
    nodes = payload.get("nodes") or []
    total = len(nodes)

    nodes_with_any = 0
    nodes_with_nonid = 0

    for nd in nodes:
        ch = (nd.get("channels") if isinstance(nd.get("channels"), dict) else nd) or {}
        tkeys = ch.get("translation") or []
        rkeys = ch.get("rotation") or []
        skeys = ch.get("scaling") or []

        any_keys = bool(tkeys or rkeys or skeys)
        if any_keys:
            nodes_with_any += 1

        # window filter if provided
        def in_win(k: dict[str, Any]) -> bool:
            t = int(k.get("time_ms", -10**9))
            if start_ms is not None and t < start_ms:
                return False
            if end_ms is not None and t > end_ms:
                return False
            return True

        nonid = False
        for k in tkeys:
            if in_win(k) and _is_non_identity_key(k, "translation"):
                nonid = True
                break
        if not nonid:
            for k in rkeys:
                if in_win(k) and _is_non_identity_key(k, "rotation"):
                    nonid = True
                    break
        if not nonid:
            for k in skeys:
                if in_win(k) and _is_non_identity_key(k, "scaling"):
                    nonid = True
                    break

        if nonid:
            nodes_with_nonid += 1

    win_str = ""
    if start_ms is not None or end_ms is not None:
        win_str = f" window=[{start_ms},{end_ms}]"

    print(f"[nodeanims]{win_str} total_nodes={total}")
    print(f"[nodeanims]{win_str} nodes_with_any_keys={nodes_with_any}")
    print(f"[nodeanims]{win_str} nodes_with_non_identity_keys={nodes_with_nonid}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mdl", type=str)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--start_ms", type=int, default=None)
    ap.add_argument("--end_ms", type=int, default=None)
    args = ap.parse_args()

    mdl_path = Path(args.mdl)
    payload = harvest_nodeanims_from_mdl(mdl_path)

    print_nodeanim_counts(payload, start_ms=args.start_ms, end_ms=args.end_ms)

    out_path = Path(args.out) if args.out else mdl_path.with_name(f"{mdl_path.stem}_nodeanims.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[nodeanims] wrote: {out_path}")


if __name__ == "__main__":
    main()
