#wc3kin/wc3mdl/import_mdl.py
from __future__ import annotations

from .parse_mdl import parse_mdl
from .build_imported import build_imported_model  # (we’ll add this next)
from .model import ImportedModel

def import_mdl(path: str) -> ImportedModel:
    """
    Standards-based importer entry point.

    - Parses MDL text to AST
    - Builds ImportedModel using canonical types in wc3kin.wc3mdl.model
    """
    ast = parse_mdl(path)
    return build_imported_model(ast)