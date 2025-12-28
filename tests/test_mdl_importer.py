import os, sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath("D:\wc3_kinematics_tool")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wc3kin.wc3mdl.parse_mdl import parse_mdl_file
from wc3kin.wc3mdl.build_model import build_model_from_raw
from wc3kin.wc3mdl.normalize_clips import normalize_sequence_clip
from wc3kin.wc3mdl.model import Model
from wc3kin.wc3mdl.eval import evaluate_pose, build_skin_matrices
from wc3kin.wc3mdl.validate import run_validation

# Path to the attached Archer.mdl
ARCHER_MDL = "D:\\wc3_all_assets\\Units\\NightElf\\Archer\\Archer.mdl"


# -------------------------------------------------------------------
# Helper for approximate matrix comparisons
# -------------------------------------------------------------------

def approx_mat4(m1, m2, tol=1e-5):
    """
    Compare two 4x4 matrices elementwise with a tolerance.
    """
    return np.allclose(np.array(m1), np.array(m2), atol=tol)


# -------------------------------------------------------------------
# Parsing tests
# -------------------------------------------------------------------

def test_parse_mdl_basic_structure():
    """
    Basic structural test for parsing – confirms essential sections are present.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    assert "Nodes" in raw, "Nodes section missing"
    assert "Sequences" in raw, "Sequences section missing"
    assert "Geosets" in raw, "Geosets section missing"

    assert isinstance(raw["Nodes"], list), "Nodes should parse to a list"
    assert isinstance(raw["Sequences"], list), "Sequences should parse to a list"


# -------------------------------------------------------------------
# Build model tests
# -------------------------------------------------------------------

def test_build_model_semantics():
    """
    Ensure semantic model building creates a Model with expected properties.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)

    assert isinstance(model, Model)
    assert model.nodes, "No nodes built from model"
    assert model.geosets, "No geosets built"
    assert model.sequences, "No sequences built"

    # Confirm that at least one root bone exists
    roots = [n for n in model.nodes.values() if n.parent_id is None]
    assert roots, "No root bones found"


# -------------------------------------------------------------------
# Clip normalization tests
# -------------------------------------------------------------------

def test_sequence_time_normalization():
    """
    Verify that key times fall within [0, dur] after normalization.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)

    for seq in model.sequences:
        normalized = normalize_sequence_clip(
            name=seq.name,
            seq_start=seq.start_abs,
            seq_end=seq.end_abs,
            raw_bone_anims=seq.raw_bone_anims,
        )

        # All keys must be in [0, seq.dur]
        for bone_id, anim in normalized.bone_anims.items():
            for key in (anim.translation or []) + (anim.rotation or []) + (anim.scaling or []):
                assert 0 <= key.t <= seq.dur, (
                    f"Normalized key outside sequence range: {key.t} in {seq.name}"
                )


# -------------------------------------------------------------------
# Evaluation tests (pose)
# -------------------------------------------------------------------

def test_bind_pose_identity_like():
    """
    Bind pose should be near identity for root bones at t=0.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)
    seq = model.sequences[0]
    normalized = normalize_sequence_clip(
        name=seq.name,
        seq_start=seq.start_abs,
        seq_end=seq.end_abs,
        raw_bone_anims=seq.raw_bone_anims,
    )

    # Evaluate world matrices at t=0
    world0 = evaluate_pose(model, normalized, t_ms=0)

    # Root bone transform ~ identity or near-bind
    roots = [n.object_id for n in model.nodes.values() if n.parent_id is None]
    for root in roots:
        m = world0[root]
        # m should be roughly identity
        assert approx_mat4(m, np.eye(4)), f"Root world matrix not identity at bind: {m}"


def test_mid_animation_consistency():
    """
    Tests that evaluation runs without error mid animation.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)
    seq = model.sequences[0]
    norm = normalize_sequence_clip(
        name=seq.name, seq_start=seq.start_abs, seq_end=seq.end_abs, raw_bone_anims=seq.raw_bone_anims
    )

    # Evaluate in the middle of the clip
    t_mid = seq.dur // 2
    pose_mid = evaluate_pose(model, norm, t_ms=t_mid)
    # All bones should have a transform matrix
    assert all(isinstance(m, list) and len(m) == 16 for m in pose_mid.values()), (
        "evaluate_pose should return flat 4x4 matrices per bone"
    )


# -------------------------------------------------------------------
# Skinning tests
# -------------------------------------------------------------------

def test_skin_matrix_t_consistency():
    """
    Tests that skin matrices undo bind-space.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)
    seq = model.sequences[0]
    norm = normalize_sequence_clip(
        name=seq.name, seq_start=seq.start_abs, seq_end=seq.end_abs, raw_bone_anims=seq.raw_bone_anims
    )

    # Build bind and mid-world
    world_bind = evaluate_pose(model, norm, t_ms=0)
    world_mid  = evaluate_pose(model, norm, t_ms=seq.dur // 2)

    inv_bind = {bid: np.linalg.inv(np.array(world_bind[bid]).reshape(4,4)).flatten().tolist()
                for bid in world_bind}

    skin = build_skin_matrices(world_mid, inv_bind)

    # Skin matrix must be near identity at bind
    # (evaluate twice at t=0)
    skin0 = build_skin_matrices(world_bind, inv_bind)
    for bid, m in skin0.items():
        assert approx_mat4(m, np.eye(4)), f"Skin at bind should be identity for {bid}"


# -------------------------------------------------------------------
# Validation tests
# -------------------------------------------------------------------

def test_validation_no_errors():
    """
    Basic validation shouldn't raise for a sane model.
    """
    raw = parse_mdl_file(ARCHER_MDL)
    model = build_model_from_raw(raw)

    # run validation and ensure it returns an empty list of errors
    errs = run_validation(model)
    assert isinstance(errs, list)
    assert not errs, f"Unexpected validation errors: {errs}"
