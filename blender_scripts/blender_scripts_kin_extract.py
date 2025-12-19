# Blender background script
# Usage (Blender):
# blender --background --python kin_extract.py -- --input "model.fbx" --output "result.json"

import argparse
import json
import os
import sys
import traceback

import bpy


SUPPORTED_IMPORTS = {".fbx", ".gltf", ".glb", ".dae", ".obj"}


def _clean_scene():
    # Do NOT read factory settings here; it can disrupt addon state and preferences.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _import_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise RuntimeError(f"Unsupported import extension: {ext}")


def _find_armature_object():
    # Prefer ARMATURE type objects
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def _bones_from_armature(arm_obj):
    if not arm_obj:
        return []
    return [b.name for b in arm_obj.data.bones]


def _action_affects_armature(action, arm_obj) -> bool:
    if not action or not arm_obj:
        return False
    # Heuristic: any fcurve targeting pose bones means it’s relevant
    for fc in action.fcurves:
        dp = fc.data_path or ""
        if dp.startswith('pose.bones["'):
            return True
    return False


def _collect_actions_for_armature(arm_obj):
    names = []
    for action in bpy.data.actions:
        if _action_affects_armature(action, arm_obj):
            names.append(action.name)
    # stable unique
    return sorted(set(names), key=lambda s: s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    result = {
        "ok": False,
        "input": args.input,
        "armature_name": None,
        "bones": [],
        "animations": [],
        "warnings": [],
        "error": None,
    }

    try:
        in_path = os.path.abspath(args.input)
        out_path = os.path.abspath(args.output)
        ext = os.path.splitext(in_path)[1].lower()

        if ext not in SUPPORTED_IMPORTS:
            result["warnings"].append(
                f"Extension {ext} not supported by this extractor. "
                f"Supported: {sorted(SUPPORTED_IMPORTS)}. "
                f"(MDL/MDX require an addon; we’ll handle that later.)"
            )
            result["ok"] = True
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return

        _clean_scene()
        _import_model(in_path)

        arm = _find_armature_object()
        if not arm:
            result["warnings"].append("No armature found (no bones/animations extracted).")
            result["ok"] = True
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return

        result["armature_name"] = arm.name
        result["bones"] = _bones_from_armature(arm)
        result["animations"] = _collect_actions_for_armature(arm)
        result["ok"] = True

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        try:
            with open(os.path.abspath(args.output), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
