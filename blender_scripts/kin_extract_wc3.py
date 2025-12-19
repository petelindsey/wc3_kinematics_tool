# blender_scripts/kin_extract_wc3.py
# blender --background --python kin_extract_wc3.py -- --input "file.mdx" --output "out.json"

import bpy
import json
import sys
import traceback
from pathlib import Path


def _argv_after_double_dash():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


def _clean_scene_like_yours():
    # match your working pattern: select all -> delete, then clear some datablocks :contentReference[oaicite:2]{index=2}
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablock in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures, bpy.data.actions):
        for x in list(datablock):
            try:
                if getattr(x, "users", 0) == 0:
                    datablock.remove(x, do_unlink=True)
            except Exception:
                pass


def _import_fbx(path: str):
    bpy.ops.import_scene.fbx(filepath=path)  # same op you use :contentReference[oaicite:3]{index=3}


def _try_import_war3(filepath: str) -> str:
    """
    Find a WC3 MDL/MDX importer operator. Addons differ, so we:
      1) try a few common guesses
      2) scan bpy.ops.* for import-ish operators containing mdl/mdx/war3/warcraft
    Returns the operator name used (e.g. "import_scene.warcraft3_mdx").
    """
    fp = str(Path(filepath).resolve())

    guesses = [
        ("import_scene", "warcraft3_mdx"),
        ("import_scene", "warcraft3_mdl"),
        ("import_scene", "warcraft_3_mdx"),
        ("import_scene", "warcraft_3_mdl"),
        ("import_scene", "war3_mdx"),
        ("import_scene", "war3_mdl"),
        ("import_scene", "mdx"),
        ("import_scene", "mdl"),
    ]

    for mod_name, op_name in guesses:
        mod = getattr(bpy.ops, mod_name, None)
        if mod and hasattr(mod, op_name):
            op = getattr(mod, op_name)
            try:
                op(filepath=fp)
                return f"{mod_name}.{op_name}"
            except Exception:
                pass

    # robust scan
    for mod_name in dir(bpy.ops):
        if mod_name.startswith("_"):
            continue
        mod = getattr(bpy.ops, mod_name, None)
        if mod is None:
            continue

        for op_name in dir(mod):
            n = op_name.lower()
            if "import" not in n:
                continue
            if not any(k in n for k in ("mdl", "mdx", "war3", "warcraft")):
                continue

            op = getattr(mod, op_name, None)
            if op is None:
                continue
            try:
                op(filepath=fp)
                return f"{mod_name}.{op_name}"
            except Exception:
                continue

    candidates = []
    for mod_name in dir(bpy.ops):
        if mod_name.startswith("_"):
            continue
        mod = getattr(bpy.ops, mod_name, None)
        if mod is None:
            continue
        for op_name in dir(mod):
            n = op_name.lower()
            if "import" not in n:
                continue
#            if "import" in n and any(k in n for k in ("mdl", "mdx", "war3", "warcraft")):
            candidates.append(f"{mod_name}.{op_name}")
    candidates = sorted(set(candidates))
    import_ops = []
    for mod_name in dir(bpy.ops):
        if mod_name.startswith("_"):
            continue
        mod = getattr(bpy.ops, mod_name, None)
        if mod is None:
            continue
        for op_name in dir(mod):
            if "import" in op_name.lower():
                import_ops.append(f"{mod_name}.{op_name}")

    addons = []
    try:
        addons = sorted([a.module for a in bpy.context.preferences.addons.values()])
    except Exception:
        pass

    import_ops = sorted(set(import_ops))

    raise RuntimeError(
        "No WC3 MDL/MDX importer operator found.\n"
        f"Enabled addons: {addons}\n"
        f"Import ops seen (first 300): {import_ops[:300]}"
    )



def _find_armature_object():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def _bones(arm_obj):
    if not arm_obj:
        return []
    try:
        return [b.name for b in arm_obj.data.bones]
    except Exception:
        return []


def _actions_affecting_armature(arm_obj):
    if not arm_obj:
        return []
    out = []
    for act in bpy.data.actions:
        try:
            for fc in act.fcurves:
                dp = fc.data_path or ""
                if dp.startswith('pose.bones["'):
                    out.append(act.name)
                    break
        except Exception:
            continue
    return sorted(set(out), key=lambda s: s.lower())


def _nla_strip_names(arm_obj):
    if not arm_obj:
        return []
    names = []
    ad = getattr(arm_obj, "animation_data", None)
    if not ad:
        return []
    tracks = getattr(ad, "nla_tracks", None)
    if not tracks:
        return []
    for tr in tracks:
        for st in tr.strips:
            if st and st.name:
                names.append(st.name)
            # also capture referenced action name if present
            act = getattr(st, "action", None)
            if act and act.name:
                names.append(act.name)
    return sorted(set(names), key=lambda s: s.lower())


def main():
    args = _argv_after_double_dash()
    if len(args) < 4:
        raise SystemExit("Usage: -- --input <file> --output <json>")

    # tiny manual parse
    in_path = None
    out_path = None
    for i in range(len(args) - 1):
        if args[i] == "--input":
            in_path = args[i + 1]
        elif args[i] == "--output":
            out_path = args[i + 1]

    if not in_path or not out_path:
        raise SystemExit("Missing --input or --output")

    in_path = str(Path(in_path).resolve())
    out_path = str(Path(out_path).resolve())
    ext = Path(in_path).suffix.lower()

    payload = {
        "ok": False,
        "input": in_path,
        "importer_used": None,
        "armature_name": None,
        "bones": [],
        "animations": [],
        "warnings": [],
        "error": None,
    }

    try:
        _clean_scene_like_yours()

        if ext == ".fbx":
            _import_fbx(in_path)
            payload["importer_used"] = "import_scene.fbx"
        elif ext in (".mdl", ".mdx"):
            payload["importer_used"] = _try_import_war3(in_path)
        else:
            payload["warnings"].append(f"Unsupported extension for extractor: {ext}")
            payload["ok"] = True
            Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        arm = _find_armature_object()
        if not arm:
            payload["warnings"].append("No armature found after import.")
            payload["ok"] = True
            Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        payload["armature_name"] = arm.name
        payload["bones"] = _bones(arm)

        anims = _actions_affecting_armature(arm)
        nla = _nla_strip_names(arm)

        # union
        payload["animations"] = sorted(set(anims + nla), key=lambda s: s.lower())
        if not payload["animations"]:
            payload["warnings"].append("No actions/NLA strips found; model may have no sequences or importer stored them differently.")

        payload["ok"] = True
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    except Exception as e:
        payload["error"] = str(e)
        payload["traceback"] = traceback.format_exc()
        try:
            Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(e)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
