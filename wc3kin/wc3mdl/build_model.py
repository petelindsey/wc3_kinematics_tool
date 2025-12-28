# wc3kin/wc3mdl/build_model.py
"""
Builds a canonical semantic model from raw MDL AST.

All normalization, time rules, and WC3 semantics live here.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

@dataclass
class BoneAnim:
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
    triangles: List[Tuple[int,int,int]]
    vertex_groups: List[int]
    matrices: List[List[int]]

@dataclass
class Model:
    nodes: Dict[int, Node]
    geosets: List[Geoset]
    sequences: Dict[str, SequenceClip]
    bind_world: Dict[int, List[List[float]]]

# ------------------ Utilities ------------------

def normalize_quat(q: Quat) -> Quat:
    x,y,z,w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return (0,0,0,1)
    return (x/n, y/n, z/n, w/n)

# ------------------ Extraction ------------------

def build_model(ast_blocks) -> Model:
    nodes = {}
    sequences = {}
    geosets = []

    pivot_points = {}

    # ---- First pass: pivots ----
    for b in ast_blocks:
        if b.type == "PivotPoints":
            for i,(_,_,v) in enumerate(b.body):
                pivot_points[i] = v

    # ---- Nodes ----
    for b in ast_blocks:
        if b.type in ("Bone","Helper","Attachment"):
            obj_id = None
            parent = None
            for stmt in b.body:
                if stmt[0] == "stmt":
                    if stmt[1].startswith("ObjectId"):
                        obj_id = int(stmt[1].split()[1])
                    if stmt[1].startswith("Parent"):
                        parent = int(stmt[1].split()[1])

            pivot = pivot_points.get(obj_id,(0,0,0))
            nodes[obj_id] = Node(obj_id, parent, pivot, b.name)

    # ---- Sequences ----
    for b in ast_blocks:
        if b.type == "Anim":
            name = b.name
            start,end = None,None
            for stmt in b.body:
                if stmt[0]=="stmt" and stmt[1].startswith("Interval"):
                    nums = [int(x) for x in stmt[1].split("{")[1].split("}")[0].split(",")]
                    start,end = nums
            sequences[name] = SequenceClip(
                name=name,
                start=start,
                end=end,
                duration=end-start
            )

    # ---- Animations ----
    for b in ast_blocks:
        if b.type in ("Bone","Helper","Attachment"):
            obj_id = next(int(s[1].split()[1]) for s in b.body if s[0]=="stmt" and s[1].startswith("ObjectId"))

            for sub in b.body:
                if not hasattr(sub,"type"):
                    continue
                if sub.type in ("Translation","Rotation","Scaling"):
                    for track in sub.body:
                        if track.type != "Anim":
                            continue
                        seq = sequences[track.name]
                        anim = seq.bone_anims.setdefault(obj_id,BoneAnim())

                        for kind,t,v in track.body:
                            # absolute time
                            if seq.start <= t <= seq.end:
                                t_rel = t - seq.start
                            elif 0 <= t <= seq.duration:
                                t_rel = t
                            else:
                                continue

                            if sub.type=="Translation":
                                anim.translation[t_rel]=v
                            elif sub.type=="Scaling":
                                anim.scale[t_rel]=v
                            elif sub.type=="Rotation":
                                anim.rotation[t_rel]=normalize_quat(v)

    # ---- Bind pose (deterministic) ----
    bind_world = extract_bind_pose(nodes)

    return Model(nodes,geosets,sequences,bind_world)
