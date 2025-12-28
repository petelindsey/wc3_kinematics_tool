# wc3kin/wc3mdl/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # canonical: (x,y,z,w)
Interp = Literal["DontInterp", "Linear", "Hermite", "Bezier"]

@dataclass(frozen=True)
class Node:
    object_id: int
    parent_id: Optional[int]
    name: str
    pivot: Vec3

@dataclass(frozen=True)
class KeyF:
    t: int  # milliseconds, ALWAYS sequence-relative inside a clip
    value: object  # Vec3 or Quat
    in_tan: object | None = None
    out_tan: object | None = None

@dataclass(frozen=True)
class Track:
    interp: Interp
    keys: List[KeyF]  # sorted by t

@dataclass(frozen=True)
class BoneAnim:
    translation: Optional[Track] = None
    rotation: Optional[Track] = None
    scaling: Optional[Track] = None

@dataclass(frozen=True)
class SequenceClip:
    name: str
    start_abs: int
    end_abs: int
    dur: int
    bone_anims: Dict[int, BoneAnim]  # object_id -> anim tracks (sequence-relative keys)

@dataclass(frozen=True)
class Geoset:
    verts: List[Vec3]
    tris: List[Tuple[int,int,int]]
    vertex_groups: List[int]                 # len = verts
    groups_matrices: Dict[int, List[int]]    # group_index -> [bone object_ids]

@dataclass(frozen=True)
class Model:
    nodes: Dict[int, Node]         # object_id -> Node
    root_ids: List[int]
    geosets: List[Geoset]
    sequences: List[SequenceClip]
