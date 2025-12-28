# wc3kin/wc3mdl/normalize_clips.py
from __future__ import annotations
from dataclasses import replace
from typing import List, Dict, Optional, Tuple
from .model import Track, KeyF, BoneAnim, SequenceClip, Quat, Vec3

def _normalize_quat_xyzw(q: Quat) -> Quat:
    x,y,z,w = q
    n = (x*x + y*y + z*z + w*w) ** 0.5
    if n == 0.0:
        return (0.0,0.0,0.0,1.0)
    return (x/n, y/n, z/n, w/n)

def _clip_track_to_sequence(
    raw_track: Track,
    seq_start: int,
    seq_end: int,
) -> Optional[Track]:
    """
    Convert raw key times into sequence-relative key times.

    Inclusion rules:
      - ABS window:  seq_start <= t_abs <= seq_end  -> t_rel = t_abs - seq_start
      - REL window:  0 <= t_rel <= dur              -> t_rel = t_abs (treated as already relative)
      - otherwise: ignore

    If both ABS-derived and REL-derived keys land on same t_rel, ABS-derived wins.
    """
    dur = seq_end - seq_start
    abs_bucket: Dict[int, KeyF] = {}
    rel_bucket: Dict[int, KeyF] = {}

    for k in raw_track.keys:
        t = k.t
        if seq_start <= t <= seq_end:
            t_rel = t - seq_start
            abs_bucket[t_rel] = KeyF(t=t_rel, value=k.value, in_tan=k.in_tan, out_tan=k.out_tan)
        elif 0 <= t <= dur:
            rel_bucket[t] = KeyF(t=t, value=k.value, in_tan=k.in_tan, out_tan=k.out_tan)

    # merge preferring abs_bucket
    merged: Dict[int, KeyF] = dict(rel_bucket)
    merged.update(abs_bucket)

    if not merged:
        return None

    keys = [merged[t] for t in sorted(merged.keys())]
    return Track(interp=raw_track.interp, keys=keys)

def normalize_sequence_clip(
    *,
    name: str,
    seq_start: int,
    seq_end: int,
    raw_bone_anims: Dict[int, BoneAnim],
) -> SequenceClip:
    dur = seq_end - seq_start
    bone_anims: Dict[int, BoneAnim] = {}

    for oid, ba in raw_bone_anims.items():
        t = _clip_track_to_sequence(ba.translation, seq_start, seq_end) if ba.translation else None
        r = _clip_track_to_sequence(ba.rotation,    seq_start, seq_end) if ba.rotation else None
        s = _clip_track_to_sequence(ba.scaling,     seq_start, seq_end) if ba.scaling else None

        # Normalize quats at import, so eval never sees non-unit.
        if r is not None:
            r_keys = [KeyF(k.t, _normalize_quat_xyzw(k.value), k.in_tan, k.out_tan) for k in r.keys]
            r = Track(interp=r.interp, keys=r_keys)

        if t or r or s:
            bone_anims[oid] = BoneAnim(translation=t, rotation=r, scaling=s)

    return SequenceClip(
        name=name,
        start_abs=seq_start,
        end_abs=seq_end,
        dur=dur,
        bone_anims=bone_anims,
    )
