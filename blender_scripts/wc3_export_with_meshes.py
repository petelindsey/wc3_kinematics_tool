#
# Run from Blender, e.g.:
#   "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" ^
#       --background ^
#       --python D:\python-extras\pyw3x\wc3_export_with_meshes.py -- ^
#       --mdx "D:\python-extras\wc3_all_assets\Units\Undead\SkeletonMage\SkeletonMage.mdx" ^
#       --model-name SkeletonMage ^
#       --json-root "D:\wc3_all_assets" ^
#       --texture-root "D:\all_textures" ^
#       --out-fbx "D:\wc3_fbx_out_test\SkeletonMage.fbx" ^
#       --test
#
# Remove --test to actually export the FBX.
#
# NOTE: You must already have your WC3 MDX importer addon installed and
# enabled in Blender (the same one you used manually).

import bpy
import sys
import json
import os

TEXTURE_ROOT_DEFAULT = r"D:\all_textures"

DEATH_VG_RATIO_THRESHOLD = 0.2


def is_plane_only_mesh(
    mesh_obj: bpy.types.Object,
    flat_ratio_threshold: float = 0.01,
    min_size: float = 1e-5,
) -> bool:
    """
    Heuristic: a mesh is considered "plane-only" (billboard / flat card) if its
    bounding-box thickness is tiny compared to its largest dimension.

    This is intentionally conservative so we don't nuke weapons or limbs:
      - If max dimension is effectively zero -> treat as degenerate and drop.
      - Otherwise, if (min_dim / max_dim) < flat_ratio_threshold -> plane-only.
    """
    if mesh_obj.type != "MESH":
        return False

    dims = mesh_obj.dimensions
    max_dim = max(dims)
    min_dim = min(dims)

    if max_dim <= min_size:
        # Completely tiny/degenerate; safe to drop
        plane_only = True
    else:
        thickness_ratio = min_dim / max_dim
        plane_only = thickness_ratio < flat_ratio_threshold

    print(
        f"[WC3-EXPORT][PLANE] {mesh_obj.name}: "
        f"dims=({dims[0]:.5f}, {dims[1]:.5f}, {dims[2]:.5f}) "
        f"-> plane_only={plane_only}"
    )
    return plane_only

def classify_mesh_deathiness(mesh):
    """
    Look at a mesh's vertex group names and decide whether it's "death-like".

    Returns:
        has_death_vg: bool  -> has at least one vg whose name contains 'death'
        is_corpse_like: bool -> fraction of death vgroups over all vgroups >= threshold
        ratio: float -> that fraction (0..1)
    """
    vg_names = [vg.name for vg in mesh.vertex_groups]
    if not vg_names:
        return False, False, 0.0

    lower = [n.lower() for n in vg_names]
    death_names = [n for n in lower if "death" in n]  # catches DeathShinL, Death    , etc.

    death_count = len(death_names)
    ratio = death_count / float(len(lower))

    has_death_vg = death_count > 0
    is_corpse_like = has_death_vg and ratio >= DEATH_VG_RATIO_THRESHOLD

    return has_death_vg, is_corpse_like, ratio

def is_death_mesh(
    mesh_obj: bpy.types.Object,
    keyword: str = "death",
    ratio_threshold: float = 0.4,
) -> bool:
    """
    Heuristic: a mesh is considered 'death/corpse' if a significant fraction of
    its vertex groups' names contain 'death' (case-insensitive).

    For SkeletonMage this correctly classifies:
      - SkeletonMage.002  -> death
      - SkeletonMage.003  -> death
      - others            -> live
    """
    vgroups = [vg.name for vg in mesh_obj.vertex_groups]
    total = len(vgroups)
    if total == 0:
        return False

    death_vgroups = [name for name in vgroups if keyword in name.lower()]
    death_count = len(death_vgroups)
    ratio = death_count / total

    is_death = death_count > 0 and ratio >= ratio_threshold

    print(
        f"[WC3-EXPORT][DEATH] {mesh_obj.name}: "
        f"death_count={death_count}/{total} (ratio={ratio:.2f}) -> is_death={is_death}"
    )
    return is_death
# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse

    parser = argparse.ArgumentParser(
        description="Import WC3 MDX and export Roblox-friendly FBX."
    )
    
    parser.add_argument(
        "--mdx",
        required=True,
        help="Path to the Warcraft 3 .mdx file.",
    )
    
    parser.add_argument(
    "--export-per-action",
    action="store_true",
    help="If set, also export one FBX per action into <Model>_anims. Default off.",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Base model name used for JSON files, e.g. 'SkeletonMage' "
            "for SkeletonMage_bones.json, SkeletonMage_geosets.json, etc."
        ),
    )
    parser.add_argument(
        "--json-root",
        required=True,
        help="Directory containing *_bones.json and *_geosets.json files.",
    )
    parser.add_argument(
        "--out-fbx",
        required=False,
        help="Output FBX path. Required unless --test is used.",
    )
    parser.add_argument(
        "--texture-root",
        required=False,
        help="Optional texture root directory (if your importer supports it).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, only prints what would be exported, no FBX file is written.",
    )
    
    parser.add_argument(
        "--actions",
        required=False,
        default="",
        help=(
            "Optional comma-separated list of action names or prefixes to export as "
            "per-action FBXs (e.g. \"Attack - 1, Walk\"). "
            "If omitted, all top-level actions are exported."
        ),
    )
    parser.add_argument(
        "--anim-manifest",
        required=False,
        default="",
        help=(
            "Optional path to write an animation manifest JSON derived from WC3 sequence metadata "
            "(names + frame ranges). This lets downstream steps inspect/ingest animations without "
            "requiring per-action FBX breakout."
        ),
    )


    parser.add_argument(
        "--drop-plane-only",
        action="store_true",
        help=(
            "If set, meshes that are effectively flat planes (billboards / "
            "cards) are dropped from both the combined FBX and the per-action FBXs."
        ),
    )

    args = parser.parse_args(argv)
    if (not args.test) and (not args.out_fbx):
        parser.error("--out-fbx is required unless --test is given")
    return args


# ---------------------------------------------------------
# Rig sanitization
# ---------------------------------------------------------

def sanitize_rig_hierarchy(armature, mesh_objs, model_name: str):
    """
    Make the hierarchy and names Roblox-friendly:

    - Ensure the armature has a unique, explicit name (e.g. 'SkeletonMage_Rig')
    - Ensure all meshes are parented directly to the armature
    - Ensure no mesh is parent of another mesh
    - Give each mesh a unique name (SkeletonMage_Mesh_1, etc.)
    """

    if armature is None:
        print("[WC3-EXPORT] sanitize_rig_hierarchy: no armature, skipping.")
        return

    rig_name = f"{model_name}_Rig"
    print(f"[WC3-EXPORT] Renaming armature '{armature.name}' -> '{rig_name}'")
    armature.name = rig_name
    if armature.data:
        armature.data.name = rig_name

    for idx, m in enumerate(mesh_objs, 1):
        new_name = f"{model_name}_Mesh_{idx}"
        print(f"[WC3-EXPORT]   Renaming mesh '{m.name}' -> '{new_name}'")
        m.name = new_name
        if m.data:
            m.data.name = new_name

        if m.parent is not armature:
            print(
                f"[WC3-EXPORT]     Reparenting '{new_name}' to '{rig_name}' "
                f"(was: {m.parent.name if m.parent else '(none)'})"
            )
            m.parent = armature

    mesh_set = set(mesh_objs)
    for m in mesh_objs:
        if m.parent in mesh_set and m.parent is not armature:
            print(
                f"[WC3-EXPORT]     Fixing mesh cycle parent for '{m.name}' "
                f"(parent was mesh '{m.parent.name}')"
            )
            m.parent = armature

    for obj in bpy.data.objects:
        if obj.type == 'EMPTY' and obj.name == model_name:
            new_empty_name = f"{model_name}_Empty"
            print(f"[WC3-EXPORT] Renaming EMPTY '{obj.name}' -> '{new_empty_name}'")
            obj.name = new_empty_name

    print("[WC3-EXPORT] Rig hierarchy sanitized.")


# ---------------------------------------------------------
# Scene prep & metadata loading
# ---------------------------------------------------------

def clear_scene():
    """Remove default cube, camera, lamp, etc."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def load_war3_metadata(json_root, model_name, mdx_path):
    """
    Load bones & geosets metadata for the given model.

    Layout:

        MDX files:
            D:\\python-extras\\wc3_all_assets\\<subdirs>\\ModelName.mdx

        JSON files:
            D:\\wc3_all_assets\\<same subdirs>\\ModelName_bones.json
            D:\\wc3_all_assets\\<same subdirs>\\ModelName_geosets.json

    This does a direct path swap based on the 'wc3_all_assets'
    segment in the MDX path and uses the 'found' mapped path.
    """

    json_root = os.path.abspath(json_root)
    mdx_path = os.path.abspath(mdx_path)

    print(f"[WC3-EXPORT] Resolving JSON for model '{model_name}'")
    print(f"[WC3-EXPORT]   MDX path:   {mdx_path}")
    print(f"[WC3-EXPORT]   JSON root:  {json_root}")

    token = "wc3_all_assets"
    lower_mdx = mdx_path.lower()
    if token not in lower_mdx:
        raise RuntimeError(
            "[WC3-EXPORT] MDX path does not contain 'wc3_all_assets', "
            "cannot map to JSON directory.\n"
            f"  MDX path: {mdx_path}"
        )

    idx = lower_mdx.index(token) + len(token)
    rel_after_token = mdx_path[idx:]  # e.g. "\Units\Undead\SkeletonMage\SkeletonMage.mdx"
    rel_after_token = rel_after_token.lstrip("\\/")

    rel_dir = os.path.dirname(rel_after_token)       # "Units\\Undead\\SkeletonMage"
    json_dir = os.path.join(json_root, rel_dir)      # "D:\\wc3_all_assets\\Units\\Undead\\SkeletonMage"

    bones_path = os.path.join(json_dir, f"{model_name}_bones.json")
    geosets_path = os.path.join(json_dir, f"{model_name}_geosets.json")

    print(f"[WC3-EXPORT]   Mapped JSON dir: {json_dir}")
    print(f"[WC3-EXPORT]   Bones JSON:      {bones_path}")
    print(f"[WC3-EXPORT]   Geosets JSON:    {geosets_path}")

    missing = []
    if not os.path.isfile(bones_path):
        missing.append(bones_path)
    if not os.path.isfile(geosets_path):
        missing.append(geosets_path)

    if missing:
        raise FileNotFoundError(
            "[WC3-EXPORT] JSON metadata files not found at mapped locations:\n  "
            + "\n  ".join(missing)
            + "\nMake sure your JSON mirrors the MDX subdirectory structure under "
            "D:\\wc3_all_assets."
        )

    with open(bones_path, "r", encoding="utf-8") as f:
        bones_data = json.load(f)
    with open(geosets_path, "r", encoding="utf-8") as f:
        geosets_data = json.load(f)

    bones = bones_data.get("bones", []) or []
    bone_names = {b.get("name") for b in bones if b.get("name")}

    geosets = geosets_data.get("geosets") or {}
    sequences = geosets_data.get("sequences") or {}
    visibility = geosets_data.get("visibility") or {}

    print(f"[WC3-EXPORT] Loaded {len(bones)} bones.")
    print(f"[WC3-EXPORT] Geosets present: {sorted(geosets.keys())}")
    print(f"[WC3-EXPORT] Sequences present: {list(sequences.keys())}")
    print(f"[WC3-EXPORT] Visibility map for geosets: {list(visibility.keys())}")

    return bone_names, geosets, sequences, visibility



def write_anim_manifest(model_name: str, sequences: dict, out_path: str):
    """
    Write a JSON manifest of WC3 sequences (animation names + frame ranges).

    We use the geosets JSON 'sequences' mapping as the source of truth rather than
    Blender Action names (which may include helper actions like Range_Nodes).
    """
    if not out_path:
        return
    if not sequences:
        print("[WC3-EXPORT] No sequences available; skipping anim manifest.")
        return

    def _parse_range(v):
        # Common shapes:
        #   {"interval": [start, end], ...}
        #   {"start": x, "end": y}
        #   [start, end]
        if isinstance(v, dict):
            if "interval" in v and isinstance(v["interval"], (list, tuple)) and len(v["interval"]) == 2:
                return int(v["interval"][0]), int(v["interval"][1])
            if "start" in v and "end" in v:
                return int(v["start"]), int(v["end"])
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
        return None, None

    items = []
    for name, meta in sequences.items():
        start, end = _parse_range(meta)
        items.append({
            "name": name,
            "start": start,
            "end": end,
        })

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "model": model_name,
        "count": len(items),
        "sequences": items,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[WC3-EXPORT] Wrote anim manifest: {out_path} ({len(items)} sequences)")


# ---------------------------------------------------------
# MDX import
# ---------------------------------------------------------

def import_mdx(mdx_path, texture_root=None):
    """
    Import a Warcraft 3 MDX into the current scene using the installed addon.
    """
    mdx_path = os.path.abspath(mdx_path)
    print(f"[WC3-EXPORT] Importing MDX: {mdx_path}")

    # Prefer explicit texture_root, otherwise fall back to hard-coded default
    texture_root = texture_root or TEXTURE_ROOT_DEFAULT

    try:
        bpy.ops.preferences.addon_enable(module="io_scene_warcraft_3")
        print("[WC3-EXPORT] Enabled addon 'io_scene_warcraft_3' (if available).")
    except Exception as e:
        print("[WC3-EXPORT] WARNING: Could not enable 'io_scene_warcraft_3' addon:", e)

    # Try to configure the addon’s prefs so it knows where textures live
    try:
        addon = bpy.context.preferences.addons.get("io_scene_warcraft_3")
        if addon is not None:
            prefs = addon.preferences
            print(f"[WC3-EXPORT] Setting WC3 addon texture root to: {texture_root}")
            for attr in ("resource_folder", "alt_resource_folder", "resource_root"):
                if hasattr(prefs, attr):
                    setattr(prefs, attr, texture_root)
                    print(f"[WC3-EXPORT]   prefs.{attr} = {getattr(prefs, attr)}")
        else:
            print("[WC3-EXPORT] WC3 addon not found in preferences; "
                  "cannot set texture root.")
    except Exception as e:
        print("[WC3-EXPORT] WARNING: Failed to set WC3 addon texture prefs:", e)

    # Find and call the import operator (same as before)
    candidates = []
    for attr in dir(bpy.ops):
        if "warcraft" in attr.lower() or "mdl" in attr.lower():
            sub = getattr(bpy.ops, attr)
            for op_name in dir(sub):
                if "mdx" in op_name.lower() or "mdl" in op_name.lower():
                    full_name = f"{attr}.{op_name}"
                    op = getattr(sub, op_name)
                    candidates.append((full_name, op))

    print("[WC3-EXPORT] Import candidates under bpy.ops:")
    for name, _ in candidates:
        print("   -", name)

    if not candidates:
        print("[WC3-EXPORT] ERROR: Could not find a WC3 MDX import operator.")
        return

    chosen_name, op = None, None
    for name, candidate in candidates:
        if name.lower() == "warcraft_3.import_mdl_mdx":
            chosen_name, op = name, candidate
            break
    if op is None:
        chosen_name, op = candidates[0]

    print(f"[WC3-EXPORT] Using importer bpy.ops.{chosen_name}")

    # Still only pass filepath – prefs handle textures
    result = op(filepath=mdx_path)
    print(f"[WC3-EXPORT] Import operator result: {result}")
    
# ---------------------------------------------------------
# Object detection (armature + meshes)
# ---------------------------------------------------------

def find_objects_to_export(bone_names):
    """
    Find the armature(s) and mesh objects that should be exported.

    Strategy:
    1) Collect all ARMATURE objects (WC3 rigs).
    2) Collect all MESH objects that:
       - Have an Armature modifier, OR
       - Are parented to an armature, OR
       - (Fallback) have a name in bone_names.
    """
    scene = bpy.context.scene
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]

    meshes = []
    armature_set = set(armatures)

    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        has_armature_mod = any(mod.type == "ARMATURE" for mod in obj.modifiers)
        parent_is_armature = obj.parent in armature_set if obj.parent else False
        name_matches_bone = obj.name in bone_names

        if has_armature_mod or parent_is_armature or name_matches_bone:
            meshes.append(obj)

    meshes = list(dict.fromkeys(meshes))

    if not armatures:
        print("[WC3-EXPORT] WARNING: No ARMATURE objects found in the scene.")
    else:
        print("[WC3-EXPORT] Armatures found:")
        for a in armatures:
            print("  -", a.name)

    print(f"[WC3-EXPORT] Meshes selected for export: {len(meshes)}")
    for m in meshes:
        arm_mod = [mod for mod in m.modifiers if mod.type == "ARMATURE"]
        arm_info = (
            f" (Armature mod: {[mod.object.name for mod in arm_mod if mod.object]})"
            if arm_mod
            else ""
        )
        parent_info = f" (Parent: {m.parent.name})" if m.parent else ""
        print(f"  - {m.name}{arm_info}{parent_info}")

    return armatures, meshes


def filter_deforming_meshes(meshes, armature):
    """Return only meshes that actually deform with the armature."""
    if armature is None:
        return meshes

    real_bones = {b.name for b in armature.data.bones}
    result = []

    for m in meshes:
        arm_mods = [mod for mod in m.modifiers if mod.type == 'ARMATURE']
        if not arm_mods:
            continue

        vg_names = {vg.name for vg in m.vertex_groups}
        if vg_names & real_bones:
            result.append(m)

    return result


# ---------------------------------------------------------
# Geoset visibility → mesh selection
# ---------------------------------------------------------

def classify_geosets_by_visibility(geosets, sequences, visibility):
    """
    Decide which geosets are 'normal' vs 'death-only' using visibility map.

    Returns:
        normal_ids: set of geoset IDs (strings) visible in any non-death seq
        death_only_ids: set of geoset IDs visible only in death seqs
    """
    if not geosets:
        return set(), set()

    if not sequences or not visibility:
        print(
            "[WC3-EXPORT] No sequence/visibility data in geosets JSON; "
            "treating all geosets as 'normal'."
        )
        return set(geosets.keys()), set()

    death_names = [name for name in sequences.keys()
                   if "death" in name.lower()]
    non_death_names = [name for name in sequences.keys()
                       if name not in death_names]

    normal_ids = set()
    death_only_ids = set()

    for gid in geosets.keys():
        vis_map = visibility.get(gid, {}) or {}

        vis_non_death = any(
            vis_map.get(seq_name, False) for seq_name in non_death_names
        )
        vis_death = any(
            vis_map.get(seq_name, False) for seq_name in death_names
        )

        if vis_death and not vis_non_death:
            death_only_ids.add(gid)
        else:
            normal_ids.add(gid)

    print("[WC3-EXPORT] Normal geosets (non-death):", sorted(normal_ids))
    print("[WC3-EXPORT] Death-only geosets:", sorted(death_only_ids))

    return normal_ids, death_only_ids


def map_geosets_to_meshes(meshes):
    """
    Map geoset IDs (as strings '0','1',...) to mesh objects.

    Assumes the importer created one mesh per geoset in a stable order.
    We use name-sorted meshes for determinism.
    """
    sorted_meshes = sorted(meshes, key=lambda m: m.name)
    mapping = {}
    for i, m in enumerate(sorted_meshes):
        gid = str(i)
        mapping[gid] = m

    print("[WC3-EXPORT] Geoset → mesh mapping (by index):")
    for gid, m in mapping.items():
        print(f"  geoset {gid} -> mesh '{m.name}'")

    return mapping


# ---------------------------------------------------------
# Action debug + per-action export
# ---------------------------------------------------------

def debug_list_actions(prefix="[WC3-EXPORT]"):
    print(f"\n{prefix} Actions currently in the scene:")
    if not bpy.data.actions:
        print(f"{prefix}   (none)")
    else:
        for act in bpy.data.actions:
            print(f"{prefix}   - {act.name}")
    print()

def get_core_actions(
    model_name: str,
    actions_filter: str = "",
):
    """
    Choose which actions to export as separate FBXs.

    Modes:
    - If actions_filter is empty:
        * Export all "top-level" actions:
          - Skip '#UNANIMATED', 'all sequences', and per-object tracks
            like 'Attack - 1 SkeletonMage.002'.
          - Skip zero-length actions.
    - If actions_filter is non-empty:
        * Treat it as a comma-separated list of names/prefixes.
        * Include any action whose name (case-insensitive):
            - equals a token, OR
            - starts with a token, OR
            - token is '*'.

      Example:
        actions_filter = "Attack - 1, Walk"
          → export only 'Attack - 1' and 'Walk' (if present).
    """
    lower_model = model_name.lower()

    tokens = []
    if actions_filter:
        tokens = [t.strip().lower() for t in actions_filter.split(",") if t.strip()]

    selected = []

    for act in bpy.data.actions:
        name = act.name
        lname = name.lower()

        # Skip helper / utility actions
        if name.startswith("#"):
            continue
        if "all sequences" in lname:
            continue

        # Skip zero-length / stub actions
        start, end = act.frame_range
        if end <= start:
            continue

        # Per-object tracks typically contain the model name
        # (e.g. 'Attack - 1 SkeletonMage.002').
        # We skip those *unless* the user explicitly asked for them.
        if not tokens and lower_model in lname:
            continue

        if tokens:
            # Explicit filter mode
            for tok in tokens:
                if tok == "*":
                    selected.append(act)
                    break
                if lname == tok or lname.startswith(tok):
                    selected.append(act)
                    break
        else:
            # Auto mode: accept any top-level action
            selected.append(act)

    print("\n[WC3-EXPORT] Core actions selected for per-action export:")
    if not selected:
        print("[WC3-EXPORT]   (none)")
    else:
        for act in selected:
            print(f"[WC3-EXPORT]   - {act.name}")

    return selected

def export_per_action_fbxs(
    model_name: str,
    armature: bpy.types.Object,
    meshes,
    out_fbx_path: str,
    actions_filter: str = "",
):
    """
    For each core action, export an FBX with only that action baked.

    Mesh selection rules (using the vertex-group-based is_death_mesh heuristic):

      - For actions whose name contains 'death' (case-insensitive):
            * Export ONLY meshes for which is_death_mesh(mesh) is True
              (corpse / death chunks).

      - For all other actions:
            * Export ONLY meshes for which is_death_mesh(mesh) is False
              (live body meshes).

    Notes:
      - The incoming 'meshes' list is assumed to have already been filtered
        for plane-only meshes (if --drop-plane-only was used).
      - We duplicate the armature OBJECT and the MESH objects, and we re-point
        their Armature modifiers to the duplicate so the FBX contains a
        self-contained rig that Roblox can recognize and skin correctly.
    """
    if armature is None:
        print("[WC3-EXPORT] No main armature; skipping per-action export.")
        return

    base_dir = os.path.dirname(out_fbx_path)
    base_name = os.path.splitext(os.path.basename(out_fbx_path))[0]
    anim_dir = os.path.join(base_dir, f"{model_name}_anims")

    if not os.path.isdir(anim_dir):
        os.makedirs(anim_dir, exist_ok=True)

    print(f"\n[WC3-EXPORT] Per-action FBX output dir: {anim_dir}")

    core_actions = get_core_actions(model_name, actions_filter)
    if not core_actions:
        print("[WC3-EXPORT] No core actions found; skipping per-action FBX export.")
        return

    scene = bpy.context.scene
    original_frame_start = scene.frame_start
    original_frame_end = scene.frame_end

    # Precompute death classification once per mesh to avoid spam & redundancy
    mesh_is_death = {}
    for m in meshes:
        mesh_is_death[m] = is_death_mesh(m)

    for act in core_actions:
        safe_action_name = act.name.replace(" ", "_").replace(":", "_")
        out_path = os.path.join(anim_dir, f"{base_name}_{safe_action_name}.fbx")
        print(f"\n[WC3-EXPORT] Exporting action '{act.name}' -> {out_path}")

        is_death_action = "death" in act.name.lower()

        # Choose which original mesh objects belong in this action export
        src_meshes = []
        dropped_corpse_meshes = []
        dropped_live_meshes = []

        for m in meshes:
            is_death = mesh_is_death.get(m, False)

            if is_death_action:
                # Death clip: keep only death/corpse meshes
                if is_death:
                    src_meshes.append(m)
                else:
                    dropped_live_meshes.append(m.name)
            else:
                # Non-death clip: keep only live meshes
                if is_death:
                    dropped_corpse_meshes.append(m.name)
                else:
                    src_meshes.append(m)

        if not src_meshes:
            print(
                f"[WC3-EXPORT]   WARNING: No meshes left after death/live filtering for '{act.name}'. Skipping."
            )
            continue

        if dropped_corpse_meshes:
            print(
                f"[WC3-EXPORT]   Dropped death/corpse meshes from non-death action '{act.name}':"
            )
            for name in dropped_corpse_meshes:
                print(f"      - {name}")

        if dropped_live_meshes:
            print(
                f"[WC3-EXPORT]   Dropped live meshes from death action '{act.name}':"
            )
            for name in dropped_live_meshes:
                print(f"      - {name}")

        # Restrict the timeline to the action's frame range
        start_frame, end_frame = act.frame_range
        scene.frame_start = int(start_frame)
        scene.frame_end = int(end_frame)

        bpy.ops.object.select_all(action="DESELECT")

        # Duplicate the armature object and its data so we have a clean, local rig
        arm_copy = armature.copy()
        arm_copy.data = armature.data.copy()
        bpy.context.collection.objects.link(arm_copy)

        # Ensure animation data is a simple single Action on the copy
        arm_copy.animation_data_create()
        if arm_copy.animation_data.nla_tracks:
            arm_copy.animation_data.nla_tracks.clear()
        arm_copy.animation_data.action = act

        # Duplicate meshes and re-point their Armature modifiers / parenting
        mesh_copies = []
        for src in src_meshes:
            mesh_copy = src.copy()
            mesh_copy.data = src.data.copy()
            bpy.context.collection.objects.link(mesh_copy)

            # If the original mesh was parented to the armature, keep that relationship
            if src.parent == armature:
                mesh_copy.parent = arm_copy

            # Make sure the Armature modifier points to the duplicate armature
            for mod in mesh_copy.modifiers:
                if mod.type == "ARMATURE":
                    mod.object = arm_copy

            mesh_copies.append(mesh_copy)

        # Select the duplicate armature and its mesh copies for export
        bpy.ops.object.select_all(action="DESELECT")
        arm_copy.select_set(True)
        for mc in mesh_copies:
            mc.select_set(True)

        bpy.context.view_layer.objects.active = arm_copy

        bpy.ops.export_scene.fbx(
            filepath=out_path,
            use_selection=True,
            apply_scale_options="FBX_SCALE_UNITS",
            bake_space_transform=True,
            bake_anim=True,
            bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            add_leaf_bones=False,
            path_mode="COPY",
            embed_textures=True,
            use_armature_deform_only=True,
            armature_nodetype="ROOT",
        )
        print(f"[WC3-EXPORT]   Done exporting '{act.name}'.")

        # Remove the duplicate armature and mesh copies
        bpy.ops.object.select_all(action="DESELECT")
        for mc in mesh_copies:
            mc.select_set(True)
        arm_copy.select_set(True)
        bpy.ops.object.delete()

    # Restore original scene frame range
    scene.frame_start = original_frame_start
    scene.frame_end = original_frame_end

def remap_image_paths_from_texture_root(texture_root: str):
    """
    If images have relative or invalid filepaths (e.g. 'SkeletonMage.png'),
    try to remap them to real files under texture_root by matching filenames.
    """
    if not texture_root:
        print("[WC3-EXPORT] No texture_root provided; skipping image remap.")
        return

    if not os.path.isdir(texture_root):
        print(f"[WC3-EXPORT] Texture root '{texture_root}' is not a directory; skipping image remap.")
        return

    print("[WC3-EXPORT] Building texture filename map from:", texture_root)

    # Build a map: lowercased filename -> absolute path
    file_map = {}
    for dirpath, dirnames, filenames in os.walk(texture_root):
        for fn in filenames:
            key = fn.lower()
            full_path = os.path.join(dirpath, fn)
            # Keep first-found for determinism
            if key not in file_map:
                file_map[key] = full_path

    print(f"[WC3-EXPORT] Texture filename map contains {len(file_map)} entries.")

    # Remap Blender image paths
    for img in bpy.data.images:
        if not img.filepath:
            continue

        # Resolve Blender's internal path to an absolute path
        current_path = bpy.path.abspath(img.filepath)
        if os.path.exists(current_path):
            # Already valid on disk; nothing to do
            continue

        base_name = os.path.basename(img.filepath).lower()
        if base_name in file_map:
            new_path = file_map[base_name]
            print(
                f"[WC3-EXPORT] Remapping image '{img.name}' "
                f"from '{img.filepath}' -> '{new_path}'"
            )
            img.filepath = new_path
        else:
            print(
                f"[WC3-EXPORT] WARNING: Could not remap image '{img.name}' "
                f"('{img.filepath}') under texture root."
            )

def select_and_export(
    model_name: str,
    armatures,
    meshes,
    out_fbx: str,
    test_only: bool = False,
):
    """
    Export the combined base rig FBX.

    - Sanitizes the hierarchy (renames armature to <Model>_Rig, meshes to
      <Model>_Mesh_i, reparents meshes).
    - Selects the armature + meshes and exports a single FBX.
    """
    # Ensure OBJECT mode
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    if not armatures and not meshes:
        print("[WC3-EXPORT] ERROR: Nothing selected to export.")
        return

    main_armature = armatures[0] if armatures else None

    print("[WC3-EXPORT] Sanitizing rig hierarchy for Roblox...")
    sanitize_rig_hierarchy(main_armature, meshes, model_name)

    # Select everything we want to export
    bpy.ops.object.select_all(action="DESELECT")
    for a in armatures:
        a.select_set(True)
    for m in meshes:
        m.select_set(True)

    active_obj = main_armature if main_armature else meshes[0]
    bpy.context.view_layer.objects.active = active_obj

    if test_only or not out_fbx:
        print("[WC3-EXPORT] Test mode enabled: no FBX will be written.")
        return

    out_dir = os.path.dirname(out_fbx)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[WC3-EXPORT] Exporting combined FBX to: {out_fbx}")

    bpy.ops.export_scene.fbx(
        filepath=out_fbx,
        use_selection=True,
        apply_scale_options="FBX_SCALE_UNITS",
        bake_space_transform=True,
        object_types={"ARMATURE", "MESH"},
        bake_anim=True,
        bake_anim_use_all_actions=True,
        bake_anim_use_nla_strips=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        add_leaf_bones=False,
        use_armature_deform_only=True,
        armature_nodetype="ROOT",
        axis_forward="-Z",
        axis_up="Y",
        path_mode="COPY",
        embed_textures=True,
    )

    print("[WC3-EXPORT] Combined FBX export complete.")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    print("[WC3-EXPORT] Starting WC3 → Roblox FBX pipeline")
    print(f"[WC3-EXPORT] MDX: {args.mdx}")
    print(f"[WC3-EXPORT] JSON root: {args.json_root}")
    print(f"[WC3-EXPORT] Model name: {args.model_name}")
    if args.texture_root:
        print(f"[WC3-EXPORT] Texture root: {args.texture_root}")

    clear_scene()

    bone_names, geosets, sequences, visibility = load_war3_metadata(
        args.json_root, args.model_name, args.mdx
    )

    # Optional: write an animation manifest derived from WC3 sequence metadata.
    # This supports downstream ingestion/inspection without per-action FBX breakout.
    if getattr(args, "anim_manifest", "") and (not args.test):
        write_anim_manifest(args.model_name, sequences, args.anim_manifest)

    # Import MDX first so images exist
    import_mdx(args.mdx, texture_root=args.texture_root)

    # Now remap and pack textures
    texture_root = args.texture_root or TEXTURE_ROOT_DEFAULT
    remap_image_paths_from_texture_root(texture_root)
    try:
        bpy.ops.file.pack_all()
        print("[WC3-EXPORT] Packed all images into the current Blender session.")
    except RuntimeError as e:
        print(f"[WC3-EXPORT] WARNING: Could not pack images: {e}")
    debug_list_actions()

    armatures, meshes = find_objects_to_export(bone_names)

    if not meshes:
        print(
            "[WC3-EXPORT] WARNING: No meshes found to export.\n"
            "Check that:\n"
            "  1) The MDX importer actually created mesh objects.\n"
            "  2) Meshes have an Armature modifier or are parented to the armature.\n"
            "  3) If necessary, adjust find_objects_to_export() to match your setup.\n"
        )
        return

    print(f"[WC3-EXPORT] Meshes selected for export (before filtering): {len(meshes)}")
    for m in meshes:
        print(f"  - {m.name}")

    # Optional: drop plane-only meshes globally (combined + per-action)
    if getattr(args, "drop_plane_only", False):
        keep_meshes = []
        dropped_planes = []
        for m in meshes:
            if is_plane_only_mesh(m):
                dropped_planes.append(m.name)
            else:
                keep_meshes.append(m)

        if dropped_planes:
            print("[WC3-EXPORT] Dropping plane-only meshes (billboards/cards):")
            for name in dropped_planes:
                print(f"  - {name}")

        meshes = keep_meshes

    main_armature = armatures[0] if armatures else None
    # Per-action exports for Roblox animation import (use filtered mesh set).
    if args.export_per_action and (not args.test) and args.out_fbx:
        export_per_action_fbxs(
            args.model_name,
            main_armature,
            meshes,
            args.out_fbx,
            actions_filter=args.actions,
        )

    # Filter non-deforming helpers from the base model
     # Filter non-deforming helpers from the base model
    meshes = filter_deforming_meshes(meshes, main_armature)

    # Optionally still use geoset visibility, when available, to strip
    # geosets that are purely death-only. This is a *pre-filter* and
    # is not strictly required for the vertex-group heuristic to work.
    normal_geosets, death_only_geosets = classify_geosets_by_visibility(
        geosets, sequences, visibility
    )
    geoset_to_mesh = map_geosets_to_meshes(meshes)

    filtered_meshes = []
    for gid, mesh in geoset_to_mesh.items():
        if gid in normal_geosets:
            filtered_meshes.append(mesh)

    # If JSON visibility info is missing / useless (e.g. no sequences),
    # fall back to just using all meshes.
    if not filtered_meshes:
        filtered_meshes = list(meshes)

    # Now apply the vertex-group-based heuristic to drop corpse/death meshes
    live_meshes = []
    corpse_mesh_names = []

    for m in filtered_meshes:
        if is_death_mesh(m):
            corpse_mesh_names.append(m.name)
        else:
            live_meshes.append(m)

    if corpse_mesh_names:
        print("[WC3-EXPORT] Dropping death/corpse meshes from base rig:")
        for name in corpse_mesh_names:
            print(f"  - {name}")

    meshes = live_meshes

    print(f"[WC3-EXPORT] Meshes selected for base model export: {len(meshes)}")

    for m in meshes:
        print(f"  - {m.name}")

    # Export combined FBX (test mode will just log)
    select_and_export(
        args.model_name,
        armatures,
        meshes,
        args.out_fbx,
        test_only=args.test,
    )


if __name__ == "__main__":
    main()
