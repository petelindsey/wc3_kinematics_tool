from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x,y,z,w)
Mat4 = list[list[float]]  # 4x4 row-major


@dataclass(frozen=True)
class SequenceDef:
    name: str
    start_ms: int
    end_ms: int
    category: str = "unknown"
    is_death: bool = False
    is_corpse: bool = False

    @property
    def duration_ms(self) -> int:
        d = int(self.end_ms) - int(self.start_ms)
        return max(d, 0)


@dataclass(frozen=True)
class BoneDef:
    object_id: int
    parent_id: Optional[int]
    name: str
    pivot: Vec3


@dataclass(frozen=True)
class KeyVec3:
    time_ms: int
    value: Vec3


@dataclass(frozen=True)
class KeyQuat:
    time_ms: int
    quat: Quat


@dataclass(frozen=True)
class BoneAnimChannels:
    object_id: int
    translation: list[KeyVec3]
    rotation: list[KeyQuat]
    scaling: list[KeyVec3]


@dataclass(frozen=True)
class Rig:
    bones: dict[int, BoneDef]
    children: dict[int, list[int]]
    roots: list[int]


@dataclass(frozen=True)
class Pose:
    world_mats: dict[int, Mat4]
    world_pos: dict[int, Vec3]
