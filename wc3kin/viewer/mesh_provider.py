#wc3kin/viewer/mesh_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional
import sqlite3
from pathlib import Path

Vec3 = tuple[float, float, float]

@dataclass(frozen=True)
class MeshData:
    # Bind-pose (model-space) vertex positions
    vertices: list[Vec3]
    # Triangle vertex indices (triples)
    triangles: list[tuple[int, int, int]]

    # Skinning (WC3 MDL-style)
    # One entry per vertex: index into `groups_matrices`
    vertex_groups: Optional[list[int]] = None
    # Each group: list of bone object_ids ("Matrices { ... }" in MDL)
    groups_matrices: Optional[list[list[int]]] = None


class MeshProvider(Protocol):
    def load_mesh(self, *, con: sqlite3.Connection, unit_id: int) -> MeshData:
        ...


class MdlFileMeshProvider:
    """
    Current path: load from the MDL on disk. Later you can switch to DB provider.
    """
    def __init__(self, *, mdl_path: Path) -> None:
        self.mdl_path = mdl_path

    def load_mesh(self, *, con: sqlite3.Connection, unit_id: int) -> MeshData:
        from .mdl_mesh_parse import parse_mdl_mesh
        return parse_mdl_mesh(self.mdl_path)


class SqliteMeshProvider:
    """
    Future path: load the MDL text (or a pre-parsed mesh payload) from sqlite,
    then parse into MeshData.
    For now this is a stub so the callsite is stable.
    """
    def __init__(self, *, table: str = "unit_mdls") -> None:
        self.table = table

    def load_mesh(self, *, con: sqlite3.Connection, unit_id: int) -> MeshData:
        # TODO: implement when DB schema exists.
        # Example idea:
        #   SELECT mdl_text FROM unit_mdls WHERE unit_id = ?
        # and then parse from string instead of file.
        raise NotImplementedError("SqliteMeshProvider not wired yet")
