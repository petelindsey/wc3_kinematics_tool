from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .sampler import sample_quat, sample_vec3
from .types import (
    BoneAnimChannels,
    BoneDef,
    KeyQuat,
    KeyVec3,
    Mat4,
    Pose,
    Quat,
    Rig,
    Vec3,
)


def mat4_identity() -> Mat4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    out = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        ar0, ar1, ar2, ar3 = a[r]
        for c in range(4):
            out[r][c] = ar0 * b[0][c] + ar1 * b[1][c] + ar2 * b[2][c] + ar3 * b[3][c]
    return out


def mat4_translate(v: Vec3) -> Mat4:
    x, y, z = v
    m = mat4_identity()
    m[0][3] = float(x)
    m[1][3] = float(y)
    m[2][3] = float(z)
    return m


def mat4_scale(v: Vec3) -> Mat4:
    x, y, z = v
    return [
        [float(x), 0.0, 0.0, 0.0],
        [0.0, float(y), 0.0, 0.0],
        [0.0, 0.0, float(z), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_from_quat(q: Quat) -> Mat4:
    # q = (x,y,z,w)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Row-major rotation matrix
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(m: Mat4, p: Vec3) -> Vec3:
    x, y, z = p
    ox = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    oy = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    oz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    return (ox, oy, oz)


def build_rig_from_bones_json(data: dict[str, Any]) -> Rig:
    roots = [int(x) for x in (data.get("roots") or [])]
    bones_arr = data.get("bones") or []

    bones: dict[int, BoneDef] = {}
    children: dict[int, list[int]] = {}

    for b in bones_arr:
        oid = int(b.get("object_id"))
        pid_raw = b.get("parent_id")
        pid = int(pid_raw) if pid_raw is not None else None
        name = str(b.get("name") or f"bone_{oid}")
        piv = b.get("pivot") or [0.0, 0.0, 0.0]
        pivot = (float(piv[0]), float(piv[1]), float(piv[2]))
        bones[oid] = BoneDef(object_id=oid, parent_id=pid, name=name, pivot=pivot)
        if pid is not None:
            children.setdefault(pid, []).append(oid)

    # Stable ordering for determinism
    for k in list(children.keys()):
        children[k].sort()

    # If roots missing, infer them
    if not roots and bones:
        all_ids = set(bones.keys())
        child_ids = {cid for kids in children.values() for cid in kids}
        roots = sorted(list(all_ids - child_ids))

    return Rig(bones=bones, children=children, roots=roots)


def _parse_vec3(v: Any, default: Vec3) -> Vec3:
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return default
    return (float(v[0]), float(v[1]), float(v[2]))


def _parse_quat(q: Any, default: Quat) -> Quat:
    if not isinstance(q, (list, tuple)) or len(q) < 4:
        return default
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def build_anims_from_boneanims_json(data: dict[str, Any]) -> dict[int, BoneAnimChannels]:
    out: dict[int, BoneAnimChannels] = {}
    arr = data.get("bones") or []
    for b in arr:
        oid = int(b.get("object_id"))
        ch = b.get("channels") or {}

        tkeys: list[KeyVec3] = []
        rkeys: list[KeyQuat] = []
        skeys: list[KeyVec3] = []

        for k in (ch.get("translation") or []):
            t = int(k.get("time_ms"))
            v = _parse_vec3(k.get("value"), (0.0, 0.0, 0.0))
            tkeys.append(KeyVec3(time_ms=t, value=v))

        for k in (ch.get("rotation") or []):
            t = int(k.get("time_ms"))
            # authoritative: quat
            q = k.get("quat")
            if q is None:
                # tolerant fallback
                q = k.get("value") or k.get("quaternion")
            qq = _parse_quat(q, (0.0, 0.0, 0.0, 1.0))
            rkeys.append(KeyQuat(time_ms=t, quat=qq))

        for k in (ch.get("scaling") or []):
            t = int(k.get("time_ms"))
            v = _parse_vec3(k.get("value"), (1.0, 1.0, 1.0))
            skeys.append(KeyVec3(time_ms=t, value=v))

        # Ensure deterministic ordering
        tkeys.sort(key=lambda kk: kk.time_ms)
        rkeys.sort(key=lambda kk: kk.time_ms)
        skeys.sort(key=lambda kk: kk.time_ms)

        out[oid] = BoneAnimChannels(object_id=oid, translation=tkeys, rotation=rkeys, scaling=skeys)

    return out


@dataclass
class UnitAnimEvaluator:
    rig: Rig
    anims: dict[int, BoneAnimChannels]

    def evaluate_pose(self, t_ms: int) -> Pose:
        # local matrices per bone
        local: dict[int, Mat4] = {}
        world: dict[int, Mat4] = {}
        world_pos: dict[int, Vec3] = {}

        # Stable iteration order
        bone_ids = sorted(self.rig.bones.keys())

        for oid in bone_ids:
            bdef = self.rig.bones[oid]
            ch = self.anims.get(oid)

            trans = sample_vec3(ch.translation, t_ms, default=(0.0, 0.0, 0.0)) if ch else (0.0, 0.0, 0.0)
            rot = sample_quat(ch.rotation, t_ms, default=(0.0, 0.0, 0.0, 1.0)) if ch else (0.0, 0.0, 0.0, 1.0)
            scale = sample_vec3(ch.scaling, t_ms, default=(1.0, 1.0, 1.0)) if ch else (1.0, 1.0, 1.0)

            # Canonical pivot convention:
            # M_local = T(pivot) * T(trans) * R(rot) * S(scale) * T(-pivot)
            pivot = bdef.pivot
            m = mat4_mul(mat4_translate(pivot), mat4_translate(trans))
            m = mat4_mul(m, mat4_from_quat(rot))
            m = mat4_mul(m, mat4_scale(scale))
            m = mat4_mul(m, mat4_translate((-pivot[0], -pivot[1], -pivot[2])))
            local[oid] = m

        # World resolve: forest roots
        def resolve(oid: int) -> None:
            if oid in world:
                return
            bdef = self.rig.bones[oid]
            if bdef.parent_id is None or bdef.parent_id not in self.rig.bones:
                world[oid] = local.get(oid, mat4_identity())
            else:
                resolve(bdef.parent_id)
                world[oid] = mat4_mul(world[bdef.parent_id], local.get(oid, mat4_identity()))

        for oid in bone_ids:
            resolve(oid)

        # Choose a consistent point per bone for drawing:
        # transform the pivot by world matrix (this matches pivot usage)
        for oid in bone_ids:
            pivot = self.rig.bones[oid].pivot
            world_pos[oid] = transform_point(world[oid], pivot)

        return Pose(world_mats=world, world_pos=world_pos)
