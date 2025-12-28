# wc3kin/wc3mdl/build_model.py
"""
Builds a canonical semantic model from raw MDL AST.

All normalization, time rules, and WC3 semantics live here.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class BoneAnim:
    translation: Dict[int, Tuple[float, float, float]]
    rotation: Dict[int, Tuple[float, float, float, float]]
    scale: Dict[int, Tuple[float, float, float]]

@dataclass
class SequenceClip:
    name: str
    start: int
    end: int
    bone_anims: Dict[int, BoneAnim]

@dataclass
class Node:
    object_id: int
    parent: int | None
    pivot: Tuple[float, float, float]

@dataclass
class Geoset:
    vertices: List[Tuple[float, float, float]]
    triangles: List[Tuple[int, int, int]]
    vertex_groups: List[int]
    matrices: List[List[int]]

@dataclass
class Model:
    nodes: Dict[int, Node]
    geosets: List[Geoset]
    sequences: List[SequenceClip]

def normalize_quat(q):
    x,y,z,w = q
    n = math.sqrt(x*x+y*y+z*z+w*w)
    if n == 0:
        return (0,0,0,1)
    return (x/n, y/n, z/n, w/n)

def build_model(ast) -> Model:
    # NOTE: This is a reference structure; real extractor would walk AST blocks
    # cleanly and deterministically.

    nodes: Dict[int, Node] = {}
    sequences: List[SequenceClip] = []
    geosets: List[Geoset] = []

    # --- Stub structure to demonstrate canonical intent ---
    # Replace AST walking logic with your actual block extraction

    return Model(nodes=nodes, geosets=geosets, sequences=sequences)
