# # wc3kin/wc3mdl/build_model.py
"""
Semantic WC3 MDL builder.

This module:
- Converts raw MDL AST into a canonical semantic model
- Applies Warcraft 3 animation rules exactly once
- Produces deterministic, analysis-safe data structures

NO evaluation or interpolation occurs here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import numpy as np

# ---------------------------------------------------------------------
# Canonical Types
# ---------------------------------------------------------------------

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

# ---------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------

@dataclass
class BoneAnim:
    """
    Sparse animation tracks.
    Keys are sequence-relative milliseconds.
    """
    translation: Dict[int, Vec3] = field(default_factory=dict)
    rotation: Dict[int, Quat] = field(default_factory=dict)
    scale: Dict[int, Vec3] = field(default_factory=dict)

@dataclass
class SequenceClip:
    name: str
    start: int
    end: int
    duration: int
    bone_anims: Dict[int, BoneAnim] = field(default_factory=dict)

@dataclass
class Node:
    object_id: int
    parent: int | None
    pivot: Vec3
    name: str

@dataclass
class Geoset:
    vertices: List[Vec3]
    triangles: List[Tuple[int, int, int]]
    vertex_groups: List[int]
    matrices: List[List[int]]

@dataclass
class Model:
    nodes: Dict[int, Node]
    geosets: List[Geoset]
    sequences: Dict[str, SequenceClip]
    bind_world: Dict[int, np.ndarray]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def normalize_quat(q: Quat) -> Quat:
    """Ensure quaternion is unit length."""
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

# ---------------------------------------------------------------------
# Deterministic Bind Pose Extraction
# ---------------------------------------------------------------------

def extract_bind_pose(nodes: Dict[int, Node]) -> Dict[int, np.ndarray]:
    """
    Build deterministic bind-pose world matrices.

    Rules:
    - No animation sampling
    - Identity local transform
    - Parent hierarchy applied
    - Pivot is ignored (bind pose is skeleton rest)
    """
    world: Dict[int, np.ndarray] = {}

    for bone_id, node in nodes.items():
        local = np.eye(4)
        if node.parent is None:
            world[bone_id] = local
        else:
            world[bone_id] = world[node.parent] @ local

    return world

# ---------------------------------------------------------------------
# Semantic Model Builder
# ---------------------------------------------------------------------

def build_model(ast_blocks) -> Model:
    """
    Convert parsed MDL AST into a canonical WC3 model.

    Time normalization rules:
    - Absolute keys in [seq.start, seq.end] → t_rel = t - seq.start
    - Keys already in [0, seq.duration] are kept
    - Keys outside both ranges are ignored
    """

    nodes: Dict[int, Node] = {}
    sequences: Dict[str, SequenceClip] = {}
    geosets: List[Geoset] = []

    pivot_points: Dict[int, Vec3] = {}

    # ------------------------------------------------------------
    # PivotPoints block
    # ------------------------------------------------------------

    for block in ast_blocks:
        if block.type == "PivotPoints":
            for idx, (_, _, value) in enumerate(block.body):
                pivot_points[idx] = value

    # ------------------------------------------------------------
    # Nodes (Bones / Helpers / Attachments)
    # ------------------------------------------------------------

    for block in ast_blocks:
        if block.type in ("Bone", "Helper", "Attachment"):
            obj_id = None
            parent = None

            for stmt in block.body:
                if stmt[0] != "stmt":
                    continue
                if stmt[1].startswith("ObjectId"):
                    obj_id = int(stmt[1].split()[1])
                elif stmt[1].startswith("Parent"):
                    parent = int(stmt[1].split()[1])

            if obj_id is None:
                continue

            pivot = pivot_points.get(obj_id, (0.0, 0.0, 0.0))
            nodes[obj_id] = Node(
                object_id=obj_id,
                parent=parent,
                pivot=pivot,
                name=block.name or f"bone_{obj_id}",
            )

    # ------------------------------------------------------------
    # Sequences
    # ------------------------------------------------------------

    for block in ast_blocks:
        if block.type == "Anim":
            start = end = None
            for stmt in block.body:
                if stmt[0] == "stmt" and stmt[1].startswith("Interval"):
                    nums = stmt[1].split("{")[1].split("}")[0]
                    start, end = (int(x) for x in nums.split(","))

            if start is None or end is None:
                continue

            sequences[block.name] = SequenceClip(
                name=block.name,
                start=start,
                end=end,
                duration=end - start,
            )

    # ------------------------------------------------------------
    # Animation Tracks
    # ------------------------------------------------------------

    for block in ast_blocks:
        if block.type not in ("Bone", "Helper", "Attachment"):
            continue

        obj_id = None
        for stmt in block.body:
            if stmt[0] == "stmt" and stmt[1].startswith("ObjectId"):
                obj_id = int(stmt[1].split()[1])
                break
        if obj_id is None:
            continue

        for sub in block.body:
            if not hasattr(sub, "type"):
                continue

            if sub.type not in ("Translation", "Rotation", "Scaling"):
                continue

            for track in sub.body:
                if track.type != "Anim":
                    continue

                seq = sequences.get(track.name)
                if seq is None:
                    continue

                anim = seq.bone_anims.setdefault(obj_id, BoneAnim())

                for kind, t_abs, value in track.body:
                    # Normalize time
                    if seq.start <= t_abs <= seq.end:
                        t = t_abs - seq.start
                    elif 0 <= t_abs <= seq.duration:
                        t = t_abs
                    else:
                        continue

                    if sub.type == "Translation":
                        anim.translation[t] = value
                    elif sub.type == "Scaling":
                        anim.scale[t] = value
                    elif sub.type == "Rotation":
                        anim.rotation[t] = normalize_quat(value)

    # ------------------------------------------------------------
    # Bind Pose
    # ------------------------------------------------------------

    bind_world = extract_bind_pose(nodes)

    return Model(
        nodes=nodes,
        geosets=geosets,
        sequences=sequences,
        bind_world=bind_world,
    )
