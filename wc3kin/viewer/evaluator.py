# wc3kin/viewer/evaluator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math


# ---------------------------
# Data structures
# ---------------------------

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x,y,z,w)
Mat4 = List[List[float]]                 # row-major, translation in [0][3],[1][3],[2][3]

from pathlib import Path
from typing import Union, Any, Dict, List, Tuple, Optional
import re



_NODE_BLOCK_TYPES = (
    "Bone", "Helper", "Attachment", "ParticleEmitter", "ParticleEmitter2",
    "RibbonEmitter", "Light", "EventObject", "CollisionShape"
)


@dataclass
class Keyframe:
    time_ms: int
    value: Any  # Vec3 or Quat


@dataclass
class AnimChannel:
    translation: List[Keyframe]
    rotation: List[Keyframe]
    scaling: List[Keyframe]


@dataclass
class Pose:
    # world_mats keyed by bone id (int); GL widget also tries str keys, but we keep ints.
    world_mats: Dict[int, Mat4]
    # Optional convenience: world positions per bone id order (index = bone id) if you want
    world_pos: List[Vec3]


@dataclass
class Rig:
    # Parent id per node id (or -1 / None if root)
    parent: Dict[int, Optional[int]]
    # Pivot point per node id
    pivot: Dict[int, Vec3]
    # Optional: list of all ids in stable traversal order
    ids: List[int]


# ---------------------------
# Matrix / quaternion helpers (row-major)
# ---------------------------

def mat4_identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_translate(x: float, y: float, z: float) -> Mat4:
    return [
        [1.0, 0.0, 0.0, float(x)],
        [0.0, 1.0, 0.0, float(y)],
        [0.0, 0.0, 1.0, float(z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_scale(x: float, y: float, z: float) -> Mat4:
    return [
        [float(x), 0.0,      0.0,      0.0],
        [0.0,      float(y), 0.0,      0.0],
        [0.0,      0.0,      float(z), 0.0],
        [0.0,      0.0,      0.0,      1.0],
    ]


def mat4_mul(A: Mat4, B: Mat4) -> Mat4:
    # row-major multiply: C = A * B
    C = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        ar0, ar1, ar2, ar3 = A[r]
        for c in range(4):
            C[r][c] = ar0 * B[0][c] + ar1 * B[1][c] + ar2 * B[2][c] + ar3 * B[3][c]
    return C


def quat_normalize(q: Quat) -> Quat:
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / n
    return (x * inv, y * inv, z * inv, w * inv)


def quat_to_mat4(q: Quat) -> Mat4:
    # row-major rotation matrix, column-vector convention (matches our mat4_mul/transform_point usage)
    x, y, z, w = quat_normalize(q)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),       0.0],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),       0.0],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy), 0.0],
        [0.0,                   0.0,                   0.0,                   1.0],
    ]


def transform_point(M: Mat4, v: Vec3) -> Vec3:
    # assumes v as (x,y,z,1) column vector; with row-major M
    x, y, z = v
    tx = M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3]
    ty = M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3]
    tz = M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3]
    return (tx, ty, tz)


# ---------------------------
# Keyframe sampling (step/hold)
# ---------------------------

def _sample_keys_hold(keys: List[Keyframe], t_ms: int, default):
    # WC3 MDL animation is typically keyframed and held; we do "last key <= t"
    if not keys:
        return default
    best = None
    for k in keys:
        if int(k.time_ms) <= int(t_ms):
            best = k
        else:
            break
    if best is None:
        return keys[0].value
    return best.value


# ---------------------------
# Public API
# ---------------------------

def build_anims_from_boneanims_json(boneanims_json: dict) -> Dict[int, AnimChannel]:
    """
    boneanims_json expected format (your existing builder likely does this already),
    but this version is defensive:
      boneanims_json["channels"][bone_id]["translation"/"rotation"/"scaling"] = [{time,value},...]
    """
    anims: Dict[int, AnimChannel] = {}

    channels = boneanims_json.get("channels") or boneanims_json.get("bones") or {}
    for k_bone, ch in channels.items():
        try:
            bone_id = int(k_bone)
        except Exception:
            continue

        tr_keys: List[Keyframe] = []
        ro_keys: List[Keyframe] = []
        sc_keys: List[Keyframe] = []

        for item in (ch.get("translation") or []):
            tr_keys.append(Keyframe(int(item.get("time", item.get("time_ms", 0))), tuple(item.get("value", (0, 0, 0)))))
        for item in (ch.get("rotation") or []):
            # rotation might be (x,y,z,w) already
            ro_keys.append(Keyframe(int(item.get("time", item.get("time_ms", 0))), tuple(item.get("value", (0, 0, 0, 1)))))
        for item in (ch.get("scaling") or []):
            sc_keys.append(Keyframe(int(item.get("time", item.get("time_ms", 0))), tuple(item.get("value", (1, 1, 1)))))

        # sort by time just in case
        tr_keys.sort(key=lambda kk: kk.time_ms)
        ro_keys.sort(key=lambda kk: kk.time_ms)
        sc_keys.sort(key=lambda kk: kk.time_ms)

        anims[bone_id] = AnimChannel(translation=tr_keys, rotation=ro_keys, scaling=sc_keys)

    return anims


def build_rig_from_mdl_nodes(nodes_by_id: Dict[int, dict]) -> Rig:
    """
    Build a Rig from mdl_nodes.load_nodes_from_mdl output.
    We assume nodes_by_id[node_id] has at least:
      - "parent_id" (or None/-1)
      - "pivot" (tuple/list 3 floats) OR "pivot_point"/"pivotPoint"
    """
    parent: Dict[int, Optional[int]] = {}
    pivot: Dict[int, Vec3] = {}

    ids = sorted(int(i) for i in nodes_by_id.keys())

    for nid in ids:
        n = nodes_by_id[nid] or {}
        p = n.get("parent_id", n.get("parentId", n.get("parent", None)))
        if p is None:
            parent[nid] = None
        else:
            try:
                pi = int(p)
                parent[nid] = pi if pi >= 0 else None
            except Exception:
                parent[nid] = None

        pv = n.get("pivot", n.get("pivot_point", n.get("pivotPoint", (0.0, 0.0, 0.0))))
        try:
            px, py, pz = pv
            pivot[nid] = (float(px), float(py), float(pz))
        except Exception:
            pivot[nid] = (0.0, 0.0, 0.0)

    return Rig(parent=parent, pivot=pivot, ids=ids)


class UnitAnimEvaluator:
    def __init__(self, *, rig: Rig, anims: Dict[int, AnimChannel]) -> None:
        self.rig = rig
        self.anims = anims

    def evaluate_pose(self, t_abs_ms: int, seq_start_ms: int, seq_dur_ms: int) -> Pose:
        """
        Local = T(pivot) * T(anim_translation) * R(anim_rotation) * S(anim_scale) * T(-pivot)
        World = ParentWorld * Local
        """
        world_mats: Dict[int, Mat4] = {}
        max_id = max(self.rig.ids) if self.rig.ids else -1
        world_pos: List[Vec3] = [(0.0, 0.0, 0.0)] * (max_id + 1 if max_id >= 0 else 0)

        t_rel_ms = int(t_abs_ms) - int(seq_start_ms)

        def local_mat_for(nid: int) -> Mat4:
            ch = self.anims.get(nid)

            tr = (0.0, 0.0, 0.0)
            ro = (0.0, 0.0, 0.0, 1.0)
            sc = (1.0, 1.0, 1.0)

            if ch is not None:
                # Decide per-channel time domain (mixed abs/rel exists in Archer)
                use_rel = False
                for ks in (ch.translation, ch.rotation, ch.scaling):
                    if ks and int(ks[-1].time_ms) <= int(seq_dur_ms) + 2:
                        use_rel = True
                        break
                t_ch = t_rel_ms if use_rel else int(t_abs_ms)

                tr = _sample_keys_linear(ch.translation, t_ch, tr)
                ro = _sample_keys_linear(ch.rotation, t_ch, ro)
                sc = _sample_keys_linear(ch.scaling, t_ch, sc)

            try:
                tx, ty, tz = (float(tr[0]), float(tr[1]), float(tr[2]))
            except Exception:
                tx, ty, tz = 0.0, 0.0, 0.0

            try:
                qx, qy, qz, qw = (float(ro[0]), float(ro[1]), float(ro[2]), float(ro[3]))
            except Exception:
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            try:
                sx, sy, sz = (float(sc[0]), float(sc[1]), float(sc[2]))
            except Exception:
                sx, sy, sz = 1.0, 1.0, 1.0

            px, py, pz = self.rig.pivot.get(nid, (0.0, 0.0, 0.0))

            Tp = mat4_translate(px, py, pz)
            Tn = mat4_translate(-px, -py, -pz)
            Tt = mat4_translate(tx, ty, tz)
            R = quat_to_mat4((qx, qy, qz, qw))
            S = mat4_scale(sx, sy, sz)

            return mat4_mul(Tt, mat4_mul(Tp, mat4_mul(R, mat4_mul(S, Tn))))

        def compute_world(nid: int, stack: Optional[set[int]] = None) -> Mat4:
            if nid in world_mats:
                return world_mats[nid]
            if stack is None:
                stack = set()
            if nid in stack:
                # cycle guard; treat as root
                w = local_mat_for(nid)
                world_mats[nid] = w
                return w
            stack.add(nid)

            local = local_mat_for(nid)
            pid = self.rig.parent.get(nid)

            if pid is None:
                world = local
            else:
                parent_world = compute_world(pid, stack)
                world = mat4_mul(parent_world, local)

            world_mats[nid] = world
            stack.remove(nid)
            return world

        # Ensure every rig bone gets a world matrix, regardless of ordering
        for nid in self.rig.ids:
            w = compute_world(nid)
            if nid < len(world_pos):
                world_pos[nid] = transform_point(w, (0.0, 0.0, 0.0))

        return Pose(world_mats=world_mats, world_pos=world_pos)

    

def build_rig_from_mdl_nodes(nodes_by_id: Dict[int, Any]) -> Rig:
    """
    Build a Rig from mdl_nodes.load_nodes_from_mdl output.

    Supports both:
      - dict-like nodes (older codepaths)
      - MdlNode dataclass objects (current mdl_nodes.py)

    Expected per node:
      - parent_id (or parentId/parent); -1 treated as None
      - pivot (or pivot_point/pivotPoint); defaults to (0,0,0)
    """

    def _get(node: Any, key: str, default: Any = None) -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get(key, default)
        return getattr(node, key, default)

    parent: Dict[int, Optional[int]] = {}
    pivot: Dict[int, Vec3] = {}

    ids = sorted(int(i) for i in nodes_by_id.keys())

    for nid in ids:
        n = nodes_by_id.get(nid)

        # --- parent ---
        p = _get(n, "parent_id", None)
        if p is None:
            p = _get(n, "parentId", None)
        if p is None:
            p = _get(n, "parent", None)

        try:
            pi = int(p) if p is not None else None
        except Exception:
            pi = None

        # WC3 uses -1 for no-parent in some exports
        if pi is None or pi < 0:
            parent[nid] = None
        else:
            parent[nid] = pi

        # --- pivot ---
        pv = _get(n, "pivot", None)
        if pv is None:
            pv = _get(n, "pivot_point", None)
        if pv is None:
            pv = _get(n, "pivotPoint", (0.0, 0.0, 0.0))

        try:
            px, py, pz = pv
            pivot[nid] = (float(px), float(py), float(pz))
        except Exception:
            pivot[nid] = (0.0, 0.0, 0.0)

    return Rig(parent=parent, pivot=pivot, ids=ids)

def _extract_balanced_block(text: str, start_idx: int) -> str:
    """Extract {...} block starting at the first '{' after start_idx (balanced braces)."""
    i = text.find("{", start_idx)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:j + 1]
    return ""

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _lerp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t))

def _quat_normalize(q: Quat) -> Quat:
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

def _quat_slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    # Basic slerp (good enough for WC3 "Linear" rotation tracks)
    x0, y0, z0, w0 = _quat_normalize(q0)
    x1, y1, z1, w1 = _quat_normalize(q1)

    dot = x0*x1 + y0*y1 + z0*z1 + w0*w1
    # take shortest path
    if dot < 0.0:
        dot = -dot
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1

    # If very close, lerp + normalize
    if dot > 0.9995:
        q = (
            _lerp(x0, x1, t),
            _lerp(y0, y1, t),
            _lerp(z0, z1, t),
            _lerp(w0, w1, t),
        )
        return _quat_normalize(q)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_t = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_0
    s1 = sin_t / sin_0
    return (
        x0*s0 + x1*s1,
        y0*s0 + y1*s1,
        z0*s0 + z1*s1,
        w0*s0 + w1*s1,
    )

def _sample_keys_linear(keys: List["Keyframe"], t_ms: int, default):
    """
    Linear interpolation between bracketing keys. Clamps outside range.
    Works for Vec3 and Quat (Quat handled separately below).
    """
    if not keys:
        return default
    if t_ms <= keys[0].time_ms:
        return keys[0].value
    if t_ms >= keys[-1].time_ms:
        return keys[-1].value

    # Find bracketing keys (linear scan is fine for small lists; can optimize later)
    k0 = keys[0]
    for i in range(1, len(keys)):
        k1 = keys[i]
        if t_ms <= k1.time_ms:
            k0 = keys[i - 1]
            dt = (k1.time_ms - k0.time_ms)
            if dt <= 0:
                return k1.value
            alpha = (t_ms - k0.time_ms) / dt
            # Dispatch based on tuple length
            if isinstance(k0.value, tuple) and len(k0.value) == 3:
                return _lerp_vec3(k0.value, k1.value, alpha)
            # For 4-tuple, assume quat and slerp
            if isinstance(k0.value, tuple) and len(k0.value) == 4:
                return _quat_slerp(k0.value, k1.value, alpha)
            # Fallback: hold
            return k0.value
    return keys[-1].value

def _find_named_block(body: str, name: str) -> str:
    """
    Find a top-level 'name { ... }' block inside `body`.
    We only want blocks like:
        Translation { ... }
        Rotation { ... }
        Scaling { ... }
    """
    # Find the first occurrence of '\n\tTranslation {' etc. (whitespace tolerant)
    #m = re.search(rf"\b{name}\b\s*\{{", body)
    m = re.search(rf"\b{name}\b(?:\s+\d+)?\s*\{{", body)
    if not m:
        return ""
    return _extract_balanced_block(body, m.start())

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_keyframes_vec(track_block: str, n: int) -> List["Keyframe"]:
    """
    Parse lines like:
        15933: { -19.58, -0.59, 24.59 },
    into Keyframe(time_ms, tuple_of_n_floats)
    """
    out: List[Keyframe] = []
    # keys typically appear as: <time>: { ... }
    for m in re.finditer(r"(\d+)\s*:\s*\{([^}]*)\}", track_block):
        t = int(m.group(1))
        vals = [float(x) for x in _num_re.findall(m.group(2))]
        if len(vals) >= n:
            out.append(Keyframe(time_ms=t, value=tuple(vals[:n])))
    out.sort(key=lambda k: k.time_ms)
    return out

def build_anims_from_mdl(mdl_path: Union[str, Path]) -> Dict[int, "AnimChannel"]:
    """
    Parse node animation tracks directly from an MDL file.
    Returns bone_id -> AnimChannel where bone_id is ObjectId from the node block.
    """
    mdl_path = Path(mdl_path)
    text = mdl_path.read_text(encoding="utf-8", errors="ignore")

    anims: Dict[int, AnimChannel] = {}

    # Scan for node blocks (Bone "X" { ... }, Helper "Y" { ... }, etc.)
    # We search for each block header, then extract balanced braces.
    for block_type in _NODE_BLOCK_TYPES:
        # Example: Bone "Arrow" {
        for m in re.finditer(rf'\b{block_type}\b\s+"[^"]*"\s*\{{', text):
            block = _extract_balanced_block(text, m.start())
            if not block:
                continue

            # ObjectId <int>,
            mo = re.search(r"\bObjectId\b\s+(\d+)\s*,", block)
            if not mo:
                continue
            bone_id = int(mo.group(1))

            # Pull tracks if present
            tr_block = _find_named_block(block, "Translation")
            ro_block = _find_named_block(block, "Rotation")
            sc_block = _find_named_block(block, "Scaling")

            tr_keys = _parse_keyframes_vec(tr_block, 3) if tr_block else []
            ro_keys = _parse_keyframes_vec(ro_block, 4) if ro_block else []
            sc_keys = _parse_keyframes_vec(sc_block, 3) if sc_block else []

            # Only store if any keys exist (otherwise leave bone with default TRS)
            if tr_keys or ro_keys or sc_keys:
                anims[bone_id] = AnimChannel(
                    translation=tr_keys,
                    rotation=ro_keys,
                    scaling=sc_keys,
                )
    # debug: print a few rotation keys
    printed = 0
    for bid, ch in anims.items():
        if ch.rotation:
            print("[rotkey]", "bone", bid, "t", ch.rotation[0].time_ms, "val", ch.rotation[0].value)
            printed += 1
            if printed >= 5:
                break

    return anims