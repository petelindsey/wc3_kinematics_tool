#fingerprints.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FingerprintConfig:
    active_eps: float = 1e-4
    w_range: float = 10.0
    w_omega: float = 2.0
    w_trans: float = 1.0
    top_k: int = 12  # store top movers for debugging / later similarity seeds


def motion_score(cfg: FingerprintConfig, *, rot_range: float, rot_rms: float, trans_rms: float) -> float:
    return (cfg.w_range * rot_range) + (cfg.w_omega * rot_rms) + (cfg.w_trans * trans_rms)


def build_sequence_fingerprint_payload(
    *,
    unit_id: int,
    unit_name: str,
    sequence_id: int,
    sequence_name: str,
    start_ms: int,
    end_ms: int,
    bone_rows: Iterable[dict[str, Any]],
    cfg: FingerprintConfig = FingerprintConfig(),
) -> dict[str, Any]:
    """
    bone_rows must include (at minimum):
      bone_object_id, bone_name,
      trans_rms_vel, rot_ang_range_rad, rot_rms_ang_vel
    """
    duration_ms = max(0, int(end_ms) - int(start_ms))

    movers: list[dict[str, Any]] = []
    active_mask: list[int] = []

    total_trans_rms = 0.0
    total_rot_rms = 0.0
    active_count = 0
    nonzero_count = 0

    for r in bone_rows:
        bone_object_id = int(r["bone_object_id"])
        bone_name = str(r.get("bone_name") or "")

        trans_rms = float(r.get("trans_rms_vel", 0.0))
        rot_range = float(r.get("rot_ang_range_rad", 0.0))
        rot_rms = float(r.get("rot_rms_ang_vel", 0.0))

        score = motion_score(cfg, rot_range=rot_range, rot_rms=rot_rms, trans_rms=trans_rms)

        total_trans_rms += trans_rms
        total_rot_rms += rot_rms

        if score > cfg.active_eps:
            active_count += 1
            active_mask.append(bone_object_id)
        
        if score > 0.0:
            nonzero_count += 1

        if score>1e-12:
            movers.append(
                {
                    "bone_object_id": bone_object_id,
                    "bone_name": bone_name,
                    "trans_rms_vel": trans_rms,
                    "rot_ang_range_rad": rot_range,
                    "rot_rms_ang_vel": rot_rms,
                    "motion_score": score,
                }
        )

    # Deterministic ordering
    # - active mask sorted ascending (stable for hashing)
    active_mask.sort()

    # - movers sorted by (score desc, object_id asc)
    movers.sort(key=lambda m: (-float(m["motion_score"]), int(m["bone_object_id"])))

    payload = {
        "v": 1,
        "unit": {"id": unit_id, "name": unit_name},
        "sequence": {
            "id": sequence_id,
            "name": sequence_name,
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "duration_ms": duration_ms,
        },
        "config": {
            "active_eps": cfg.active_eps,
            "w_range": cfg.w_range,
            "w_omega": cfg.w_omega,
            "w_trans": cfg.w_trans,
            "top_k": cfg.top_k,
        },
        "aggregate": {
            "active_bone_count": int(active_count),
            "nonzero_bone_count": int(nonzero_count),  
            "total_trans_rms": float(total_trans_rms),
            "total_rot_rms": float(total_rot_rms),
        },
        "active_bone_mask": active_mask,
        "top_movers": movers[: int(cfg.top_k)],
    }
    return payload


def canonicalize_fingerprint(payload: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (canonical_json, sha1_hex) deterministically.
    """
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sha1_hex = hashlib.sha1(canonical_json.encode("utf-8")).hexdigest()
    return canonical_json, sha1_hex
