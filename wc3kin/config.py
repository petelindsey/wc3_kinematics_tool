#config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

CONFIG_FIELDS = [
    ("units_root", Path),
    ("db_path", Path),
    ("model_extensions_priority", lambda v: [str(x).lower() for x in v], []),
    ("log_path", Path, lambda: Path("logs/app.log")),
    ("blender_path", Path, None),

    # Step 2.1: WC3 conversion support (mdl/mdx -> fbx cache)
    ("wc3_json_root", Path, None),      # e.g. D:\wc3_all_assets
    ("texture_root", Path, None),       # e.g. D:\all_textures
]


@dataclass(frozen=True)
class AppConfig:
    units_root: Path
    db_path: Path
    model_extensions_priority: List[str]
    log_path: Path
    blender_path: Optional[Path]
    wc3_json_root: Optional[Path]
    texture_root: Optional[Path]


def load_config(config_path: Path) -> AppConfig:
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    values = {}
    for entry in CONFIG_FIELDS:
        key = entry[0]
        cast = entry[1]
        default = entry[2] if len(entry) > 2 else None

        if key in raw:
            value = raw[key]
        else:
            value = default() if callable(default) else default

        if value is not None and cast is not None:
            value = cast(value)

        values[key] = value

    return AppConfig(**values)
