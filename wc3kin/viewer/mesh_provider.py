#wc3kin/viewer/mesh_provider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional
import sqlite3
from pathlib import Path
import re

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class MeshData:
    vertices: list[Vec3]
    triangles: list[tuple[int, int, int]]

    vertex_groups: Optional[list[int]] = None
    groups_matrices: Optional[list[list[int]]] = None

    uvs: Optional[list[tuple[float, float]]] = None

    # Optional simple fallback (still useful)
    texture_name: Optional[str] = None

    # NEW: material selection for THIS geoset (currently geoset[0])
    geoset_material_id: Optional[int] = None

    # NEW: textures table extracted from MDL Bitmaps
    # each entry: {"image": "..."} or {"replaceable_id": 1}
    textures: Optional[list[dict]] = None

    # materials table extracted from MDL:
    # each material: {"layers": [{"texture_id": 0, "filter_mode": "...", "alpha": 1.0, ...}, ...]}
    materials: Optional[list[dict]] = None

    # If present, this MeshData is a wrapper and actual renderable geosets are here
    submeshes: Optional[list["MeshData"]] = None


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
        from .mdl_mesh_parse import parse_mdl_mesh_all
        meshes = parse_mdl_mesh_all(self.mdl_path)
        if not meshes:
            raise RuntimeError(f"No geosets parsed from {self.mdl_path}")
        if len(meshes) == 1:
            return meshes[0]
        # Wrap multiple geosets so renderer/UI can toggle them
        return MeshData(vertices=[], triangles=[], submeshes=list(meshes))


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
