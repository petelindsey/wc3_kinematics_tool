#wc3kin/wc3mdl/build_imported.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any

from .parse_mdl import Block
from .model import (
    ImportedModel,
    Node,
    SequenceClip,
    BoneAnim,
    Track,
    KeyF,
    Geoset,
    GeosetAnim,
    Material,
)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _extract_global_sequences(ast: List[Block]) -> List[int]:
    # GlobalSequences { Duration { 1000, 2000, ... } }
    for b in ast:
        if b.type != "GlobalSequences":
            continue
        # find inner Duration block if present
        for sub in b.body:
            if isinstance(sub, Block) and sub.type == "Duration":
                durs = []
                for s in _stmt_lines(sub.body):
                    vals = _ints(s)
                    durs.extend(vals)
                return durs
        # fallback: durations directly as stmt lines
        durs = []
        for s in _stmt_lines(b.body):
            vals = _ints(s)
            durs.extend(vals)
        return durs
    return []

def geosets_touched_by_sequence(m: ImportedModel, seq_name: str) -> List[int]:
    seq = next((s for s in m.sequences if s.name == seq_name), None)
    if not seq:
        return []
    touched = []
    for gid, ga in m.geoset_anims.items():
        tr = ga.alpha or ga.visibility
        if not tr:
            continue
        if any(0 <= k.t <= seq.dur for k in tr.keys):
            touched.append(gid)
    return sorted(set(touched))

def _merge_tracks(a: Optional[Track], b: Track) -> Track:
    if a is None:
        return b
    # Prefer interpolation from 'a' unless you want to enforce consistency
    keys = list(a.keys) + list(b.keys)
    keys.sort(key=lambda k: k.t)
    return Track(interp=a.interp, keys=keys)

def _collect_key_lines(body: List[Any]) -> List[Tuple[int, Any]]:
    out: List[Tuple[int, Any]] = []
    for item in body:
        if isinstance(item, tuple) and item and item[0] == "key":
            _, t, val = item
            out.append((t, val))
    return out

def _interp_from_body(body: List[Any]) -> str:
    for s in _stmt_lines(body):
        if s in ("DontInterp", "Linear", "Hermite", "Bezier"):
            return s
    return "Linear"

def _normalize_time(t_abs: int, seq_start: int, seq_end: int) -> Optional[int]:
    dur = seq_end - seq_start
    if seq_start <= t_abs <= seq_end:
        return t_abs - seq_start
    if 0 <= t_abs <= dur:
        return t_abs
    return None

def _track_from_anim_block(anim_block: Block, seq_start: int, seq_end: int) -> Optional[Track]:
    keys: List[KeyF] = []
    for t_abs, value in _collect_key_lines(anim_block.body):
        t_rel = _normalize_time(t_abs, seq_start, seq_end)
        if t_rel is None:
            continue
        keys.append(KeyF(t=t_rel, value=value))
    if not keys:
        return None
    keys.sort(key=lambda k: k.t)
    return Track(interp=_interp_from_body(anim_block.body), keys=keys)

def _nums(line: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(line)]

def _ints(line: str) -> List[int]:
    return [int(float(x)) for x in _NUM_RE.findall(line)]

def _braced_tuple3(line: str) -> Optional[Tuple[float, float, float]]:
    if "{" not in line or "}" not in line:
        return None
    vals = _nums(line)
    if len(vals) >= 3:
        return (vals[0], vals[1], vals[2])
    return None

def _stmt_lines(body: List[Any]) -> List[str]:
    return [x[1] for x in body if isinstance(x, tuple) and x and x[0] == "stmt"]

def _extract_pivots(ast: List[Block]) -> Dict[int, Tuple[float, float, float]]:
    pivots: Dict[int, Tuple[float, float, float]] = {}
    for b in ast:
        if b.type != "PivotPoints":
            continue
        idx = 0
        for s in _stmt_lines(b.body):
            t = _braced_tuple3(s)
            if t is not None:
                pivots[idx] = t
                idx += 1
    return pivots

def _extract_node_anims(ast: List[Block], sequences: List[SequenceClip]) -> List[SequenceClip]:
    """
    Extract Bone/Helper/Attachment TRS tracks and attach them to sequences.

    MDL stores node tracks like:
      Bone "X" { ObjectId N, Translation { ... keys ... }, Rotation { ... }, Scaling { ... } }

    Keys are often authored on the global timeline (absolute), so we slice by each
    sequence Interval [start_abs, end_abs] and normalize to sequence-relative t=0..dur.
    """
    # one dict per sequence: object_id -> BoneAnim
    per_seq: List[Dict[int, BoneAnim]] = [dict() for _ in sequences]

    for b in ast:
        if b.type not in ("Bone", "Helper", "Attachment"):
            continue

        obj_id: Optional[int] = None
        for s in _stmt_lines(b.body):
            if s.startswith("ObjectId"):
                vals = _ints(s)
                if vals:
                    obj_id = int(vals[0])
                    break
        if obj_id is None:
            continue

        # Collect track blocks under this node
        track_blocks: List[Block] = []
        for sub in b.body:
            if isinstance(sub, Block) and sub.type in ("Translation", "Rotation", "Scaling"):
                track_blocks.append(sub)

        if not track_blocks:
            continue

        for si, seq in enumerate(sequences):
            cur = per_seq[si].get(obj_id, BoneAnim())

            for tb in track_blocks:
                tr = _track_from_anim_block(tb, seq.start_abs, seq.end_abs)
                if tr is None:
                    continue

                if tb.type == "Translation":
                    cur = BoneAnim(
                        translation=_merge_tracks(cur.translation, tr),
                        rotation=cur.rotation,
                        scaling=cur.scaling,
                    )
                elif tb.type == "Rotation":
                    cur = BoneAnim(
                        translation=cur.translation,
                        rotation=_merge_tracks(cur.rotation, tr),
                        scaling=cur.scaling,
                    )
                elif tb.type == "Scaling":
                    cur = BoneAnim(
                        translation=cur.translation,
                        rotation=cur.rotation,
                        scaling=_merge_tracks(cur.scaling, tr),
                    )

            # Only store if we actually have any channel
            if cur.translation or cur.rotation or cur.scaling:
                per_seq[si][obj_id] = cur

    # Rebuild SequenceClip list (SequenceClip is frozen)
    out: List[SequenceClip] = []
    for si, seq in enumerate(sequences):
        out.append(
            SequenceClip(
                name=seq.name,
                start_abs=seq.start_abs,
                end_abs=seq.end_abs,
                dur=seq.dur,
                bone_anims=per_seq[si],
            )
        )
    return out



def _extract_nodes(ast: List[Block], pivots: Dict[int, Tuple[float, float, float]]) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    for b in ast:
        if b.type not in ("Bone", "Helper", "Attachment"):
            continue
        obj_id = None
        parent_id: Optional[int] = None
        for s in _stmt_lines(b.body):
            if s.startswith("ObjectId"):
                obj_id = int(s.split()[1])
            elif s.startswith("Parent"):
                parent_id = int(s.split()[1])
        if obj_id is None:
            continue
        nodes[obj_id] = Node(
            object_id=obj_id,
            parent_id=parent_id,
            name=b.name or f"{b.type}_{obj_id}",
            pivot=pivots.get(obj_id, (0.0, 0.0, 0.0)),
        )
    return nodes

def _root_ids(nodes: Dict[int, Node]) -> List[int]:
    roots = [nid for nid, n in nodes.items() if n.parent_id is None or n.parent_id not in nodes]
    roots.sort()
    return roots

def _walk_blocks(items: List[Any]):
    """Depth-first walk of Blocks in an AST list (including nested)."""
    for it in items:
        if isinstance(it, Block):
            yield it
            yield from _walk_blocks(it.body)


def _extract_sequences(ast: List[Block]) -> List[SequenceClip]:
    """
    Extract animation SequenceClip entries.

    WC3 MDL structure:
      Sequences N { Anim "Name" { Interval { start, end }, ... } ... }

    Older/odd files may also place Anim blocks elsewhere; we fall back to a
    recursive scan if no Sequences block is present.
    """
    seqs: List[SequenceClip] = []

    # Preferred: Anim blocks inside Sequences { ... }
    seq_blocks = [b for b in ast if b.type == "Sequences"]
    anim_blocks: List[Block] = []
    if seq_blocks:
        for sb in seq_blocks:
            anim_blocks.extend(
                [x for x in sb.body if isinstance(x, Block) and x.type == "Anim"]
            )
    else:
        # Fallback: scan all blocks depth-first
        anim_blocks = [b for b in _walk_blocks(ast) if b.type == "Anim"]

    for b in anim_blocks:
        start = end = None
        for s in _stmt_lines(b.body):
            if s.startswith("Interval"):
                vals = _ints(s)
                if len(vals) >= 2:
                    start, end = vals[0], vals[1]
                    break

        if start is None or end is None:
            continue

        seqs.append(
            SequenceClip(
                name=b.name or "Anim",
                start_abs=start,
                end_abs=end,
                dur=max(0, end - start),
                bone_anims={},  # filled later
            )
        )

    return seqs

def _extract_geosets(ast: List[Block]) -> List[Geoset]:
    out: List[Geoset] = []
    gid = 0

    for b in ast:
        if b.type != "Geoset":
            continue

        verts: List[Tuple[float, float, float]] = []
        tris: List[Tuple[int, int, int]] = []
        vgroups: List[int] = []
        groups_matrices: Dict[int, List[int]] = {}
        material_id: Optional[int] = None

        for s in _stmt_lines(b.body):
            if s.startswith("MaterialID"):
                vals = _ints(s)
                if vals:
                    material_id = vals[0]

        for sub in b.body:
            if not isinstance(sub, Block):
                continue

            if sub.type == "Vertices":
                for s in _stmt_lines(sub.body):
                    t = _braced_tuple3(s)
                    if t is not None:
                        verts.append(t)

            elif sub.type == "VertexGroup":
                for s in _stmt_lines(sub.body):
                    s2 = s.strip()
                    if s2:
                        try:
                            vgroups.append(int(s2))
                        except ValueError:
                            pass

            elif sub.type == "Faces":
                for faces_sub in sub.body:
                    if not isinstance(faces_sub, Block):
                        continue
                    if faces_sub.type != "Triangles":
                        continue
                    for s in _stmt_lines(faces_sub.body):
                        if "{" in s and "}" in s:
                            idxs = _ints(s)
                            for i in range(0, len(idxs) - 2, 3):
                                tris.append((idxs[i], idxs[i + 1], idxs[i + 2]))

            elif sub.type == "Groups":
                group_index = 0
                for s in _stmt_lines(sub.body):
                    if s.startswith("Matrices") and "{" in s and "}" in s:
                        mats = _ints(s)
                        groups_matrices[group_index] = mats
                        group_index += 1

        out.append(
            Geoset(
                geoset_id=gid,
                verts=verts,
                tris=tris,
                vertex_groups=vgroups,
                groups_matrices=groups_matrices,
                normals=None,
                uvs=None,
                material_id=material_id,
            )
        )
        gid += 1

    return out

def _extract_geoset_anims(ast: List[Block], sequences: List[SequenceClip]) -> Dict[int, GeosetAnim]:
    out: Dict[int, GeosetAnim] = {}
    
    global_loop = False
    global_duration: Optional[int] = None

    for b in ast:
        if b.type != "GeosetAnim":
            continue

        geoset_id: Optional[int] = None
        for s in _stmt_lines(b.body):
            if s.lower().startswith("geosetid"):
                vals = _ints(s)
                if vals:
                    geoset_id = vals[0]
                    break
        if geoset_id is None:
            continue

        alpha_track: Optional[Track] = None
        global_seq_id: Optional[int] = None



        for sub in b.body:
            if not isinstance(sub, Block) or sub.type != "Alpha":
                continue

            abs_keys = _collect_key_lines(sub.body)  # [(t, value)]
            if not abs_keys:
                continue

            interp = _interp_from_body(sub.body)

            # If Alpha has a numeric tag (Alpha 1 { ... }), treat as global sequence id
            if sub.tag is not None:
                global_seq_id = int(sub.tag)

                # Keep times as-is for now; decide if it should loop
                t_list = [int(t) for (t, _) in abs_keys]
                max_t = max(t_list) if t_list else 0

                # Heuristic:
                # - many keys + "small" timeline => likely looping
                # - huge single timestamp => absolute one-shot switch
                global_loop = (len(t_list) >= 3 and max_t <= 60000)
                global_duration = (max_t + 1) if global_loop else None

                keys = [KeyF(t=int(t), value=v) for (t, v) in abs_keys]
                keys.sort(key=lambda k: k.t)
                alpha_track = Track(interp=interp, keys=keys)
                break  # IMPORTANT: stop after Alpha handled

            # Otherwise: (optional) sequence-local Alpha keys—handle later if needed
            keys = [KeyF(t=int(t), value=v) for (t, v) in abs_keys]
            keys.sort(key=lambda k: k.t)
            alpha_track = Track(interp=interp, keys=keys)

        out[geoset_id] = GeosetAnim(
        geoset_id=geoset_id,
        alpha=alpha_track,
        visibility=None,
        global_seq_id=global_seq_id,
        global_loop=global_loop,
        global_duration=global_duration,
    )

    return out


def _extract_global_sequences(ast: List[Block]) -> List[int]:
    for b in ast:
        if b.type != "GlobalSequences":
            continue

        # Common: GlobalSequences { Duration { ... } }
        for sub in b.body:
            if isinstance(sub, Block) and sub.type == "Duration":
                durs: List[int] = []
                for s in _stmt_lines(sub.body):
                    durs.extend(_ints(s))
                return durs

        # Fallback: durations directly under GlobalSequences
        durs: List[int] = []
        for s in _stmt_lines(b.body):
            durs.extend(_ints(s))
        return durs

    return []

def build_imported_model(ast: List[Block]) -> ImportedModel:
    pivots = _extract_pivots(ast)
    nodes = _extract_nodes(ast, pivots)
    roots = _root_ids(nodes)
    sequences = _extract_sequences(ast)
    sequences = _extract_node_anims(ast, sequences)
    geosets = _extract_geosets(ast)

    geoset_anims = _extract_geoset_anims(ast, sequences)
    global_sequences = _extract_global_sequences(ast)

    # NEW (whether to modulo time)
    
    return ImportedModel(
        nodes=nodes,
        root_ids=roots,
        geosets=geosets,
        sequences=sequences,
        geoset_anims=geoset_anims,
        materials=[],      
    )
