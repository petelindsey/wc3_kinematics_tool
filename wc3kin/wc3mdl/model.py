# wc3kin/wc3mdl/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Union

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # canonical: (x, y, z, w)
Interp = Literal["DontInterp", "Linear", "Hermite", "Bezier"]

KeyValue = Union[Vec3, Quat, float, int]


@dataclass(frozen=True)
class Node:
    object_id: int
    parent_id: Optional[int]
    name: str
    pivot: Vec3


@dataclass(frozen=True)
class KeyF:
    """
    Keyframe.

    IMPORTANT: t is sequence-relative milliseconds *inside a SequenceClip*.
    """
    t: int
    value: KeyValue
    in_tan: Optional[KeyValue] = None
    out_tan: Optional[KeyValue] = None


@dataclass(frozen=True)
class Track:
    interp: Interp
    keys: List[KeyF]  # sorted by t


@dataclass(frozen=True)
class BoneAnim:
    """
    Generic TRS container (name retained for compatibility).
    Can be used for bones, helpers, attachments, etc.
    """
    translation: Optional[Track] = None
    rotation: Optional[Track] = None
    scaling: Optional[Track] = None


@dataclass(frozen=True)
class SequenceClip:
    """
    One animation sequence/clip.

    start_abs/end_abs are MDL-authored absolute timeline bounds.
    bone_anims MUST store sequence-relative keys (t=0..dur) for this clip.
    """
    name: str
    start_abs: int
    end_abs: int
    dur: int
    bone_anims: Dict[int, BoneAnim]  # object_id -> TRS tracks (sequence-relative keys)


@dataclass(frozen=True)
class Geoset:
    geoset_id: int
    verts: List[Vec3]
    tris: List[Tuple[int, int, int]]

    # skinning (classic SD)
    vertex_groups: List[int]                 # len = verts
    groups_matrices: Dict[int, List[int]]    # group_index -> [bone object_ids]

    # rendering correctness (optional but recommended)
    normals: Optional[List[Vec3]] = None
    uvs: Optional[List[Tuple[float, float]]] = None
    material_id: Optional[int] = None


@dataclass(frozen=True)
class GeosetAnim:
    """
    WC3 visual animation targeting a geoset (visibility/alpha, etc.).

    geoset_id should match Geoset.geoset_id (file order index).
    Keys should follow the same convention as Track/KeyF:
      - sequence-relative t inside a SequenceClip
      - value as float/int depending on channel
    """
    geoset_id: int
    alpha: Optional[Track] = None
    visibility: Optional[Track] = None
    global_seq_id: Optional[int] = None
    global_duration: Optional[int] = None     
    global_loop: bool = False                


@dataclass(frozen=True)
class MaterialLayer:
    texture_id: Optional[int] = None
    alpha: Optional[Track] = None


@dataclass(frozen=True)
class Material:
    layers: List[MaterialLayer]


# Keep Model if you already use it elsewhere; it remains valid.
@dataclass(frozen=True)
class Model:
    nodes: Dict[int, Node]         # object_id -> Node
    root_ids: List[int]
    geosets: List[Geoset]
    sequences: List[SequenceClip]


@dataclass(frozen=True)
class ImportedModel:
    """
    Clean, viewer-ready import result.

    Uses ONLY established names from this module to avoid type ambiguity.
    The importer is responsible for normalizing MDL quirks so the viewer
    can consume this without MDL-specific logic.
    """
    nodes: Dict[int, Node]                 # all object IDs (Bone/Helper/Attachment/etc.)
    root_ids: List[int]
    geosets: List[Geoset]                  # geometry + skinning data
    sequences: List[SequenceClip]          # animation clips
    geoset_anims: Dict[int, GeosetAnim]    # geoset_id -> anim channels
    materials: List[Material]              # enough to render textures + layers
