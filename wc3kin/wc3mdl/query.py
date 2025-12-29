# wc3kin/wc3mdl/query.py
from __future__ import annotations

from typing import List, Optional
from wc3kin.wc3mdl.model import Track


def eval_track_step(track: Track, t: int):
    """
    Step evaluation: return the last key value with key.t <= t.
    Good enough for alpha/visibility gating and change detection.
    """
    last = None
    for k in track.keys:
        if k.t <= t:
            last = k.value
        else:
            break
    return last


def eval_geoset_alpha(ga, t_abs: int) -> float:
    """
    Evaluate GeosetAnim alpha at absolute time t_abs.

    WC3 default: if no Alpha key has been reached yet, alpha is 1.0 (visible).
    """
    if not getattr(ga, "alpha", None):
        return 1.0

    v = eval_track_step(ga.alpha, t_abs)

    # IMPORTANT: default alpha in WC3 is 1.0 if there is no key <= t
    if v is None:
        return 1.0

    return float(v)

def dump_seq_geoset_alpha(
    model,
    seq_names=("Death", "Decay Flesh", "Decay Bone"),
    *,
    eps: float = 0.01,
):
    """
    Debug helper: print geoset alpha at start/end of sequences.

    - Uses absolute sequence times
    - Respects missing GeosetAnim blocks (defaults to alpha=1.0)
    - Marks hidden-at-end geosets

    Intended for REPL / diagnostics, not rendering.
    """
    name_to_seq = {s.name: s for s in model.sequences}

    for seq_name in seq_names:
        if seq_name not in name_to_seq:
            print(f"\n=== {seq_name} (NOT FOUND) ===")
            continue

        s = name_to_seq[seq_name]
        print(f"\n=== {seq_name}  ({s.start_abs}..{s.end_abs}) ===")

        for geoset_id in range(len(model.geosets)):
            ga = model.geoset_anims.get(geoset_id)

            if ga is None:
                a0 = a1 = 1.0
            else:
                a0 = eval_geoset_alpha(ga, s.start_abs)
                a1 = eval_geoset_alpha(ga, s.end_abs)

            hidden = a1 <= eps
            flag = " HIDDEN" if hidden else ""
            print(
                f"geoset {geoset_id}: "
                f"alpha@start={a0:.3f}  alpha@end={a1:.3f}{flag}"
            )


def geosets_affected_in_sequence(model, seq_name: str, *, eps: float = 0.01) -> List[int]:
    """
    Returns geoset ids that are either:
      - changing alpha during the sequence (start != end), OR
      - non-default at start/end (alpha not ~1.0)

    This is often what you want for "which geosets participate in this sequence's visibility".
    """
    seq = next((s for s in model.sequences if s.name == seq_name), None)
    if seq is None:
        return []

    affected = []
    for geoset_id in range(len(model.geosets)):
        ga = model.geoset_anims.get(geoset_id)
        a0 = 1.0 if ga is None else eval_geoset_alpha(ga, seq.start_abs)
        a1 = 1.0 if ga is None else eval_geoset_alpha(ga, seq.end_abs)

        changing = abs(a0 - a1) > eps
        nondefault = (a0 < 1.0 - eps) or (a1 < 1.0 - eps)

        if changing or nondefault:
            affected.append(geoset_id)

    return affected


def geosets_hidden_at_end(m, seq_name: str, threshold: float = 0.01) -> List[int]:
    """
    Return geoset IDs whose alpha <= threshold at the end of the sequence.
    Useful for death/decay classification.
    """
    seq = next((s for s in m.sequences if s.name == seq_name), None)
    if not seq:
        return []

    t_end = int(seq.end_abs)
    hidden = []
    for gid, ga in m.geoset_anims.items():
        a = eval_geoset_alpha(ga, t_end)
        if a is not None and a <= threshold:
            hidden.append(gid)

    return sorted(hidden)
