# wc3kin/wc3mdl/validate.py
"""
Validation and deterministic diagnostics for WC3 models.
"""

import math

def validate_quaternions(model):
    for seq in model.sequences:
        for bone, anim in seq.bone_anims.items():
            for t,q in anim.rotation.items():
                n = math.sqrt(sum(c*c for c in q))
                if abs(n - 1.0) > 1e-4:
                    print(f"[WARN] Non-unit quat bone={bone} t={t} n={n}")

def validate_geosets(model):
    for g in model.geosets:
        for group in g.vertex_groups:
            if group >= len(g.matrices):
                raise ValueError("VertexGroup index out of range")

def validate_track_coverage(model):
    for seq in model.sequences:
        for bone, anim in seq.bone_anims.items():
            if not anim.rotation:
                print(f"[INFO] Bone {bone} has no rotation in {seq.name}")

def run_all(model):
    validate_quaternions(model)
    validate_geosets(model)
    validate_track_coverage(model)
