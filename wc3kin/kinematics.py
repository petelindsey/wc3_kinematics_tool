#kinematics.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import (
    HarvestJsonError,
    ingest_known_harvested_json_blobs,
    get_harvested_json_text,
    get_sequences_rows_for_unit,
    upsert_bones_by_object_id,
    upsert_bone_motion_stats,
    upsert_sequence_motion_stats,
)
from .motion_features import (
    Key3, KeyQ,
    sample_vec3, sample_quat,
    slice_times, v3_sub, v3_len, quat_angle, rms, q_norm
)
from .db import (
    get_bone_stats_for_sequence,
    upsert_sequence_fingerprint,
)
from .fingerprints import (
    FingerprintConfig,
    build_sequence_fingerprint_payload,
    canonicalize_fingerprint,
)


@dataclass(frozen=True)
class KinematicsIngestResult:
    bones_upserted: int
    motion_rows_upserted: int
    seq_rows_upserted: int


def _parse_bones_json(txt: str) -> list[dict[str, Any]]:
    data = json.loads(txt)
    bones = data.get("bones") or []
    if not isinstance(bones, list):
        return []
    return bones


def _compute_depth_and_path(parent_by_obj: dict[int, int | None], obj_id: int) -> tuple[int, str]:
    # forest + cycle-safe
    seen: set[int] = set()
    chain: list[int] = []
    cur = obj_id
    while True:
        if cur in seen:
            # cycle: break deterministically
            chain = [obj_id]
            break
        seen.add(cur)
        chain.append(cur)
        p = parent_by_obj.get(cur)
        if p is None:
            break
        cur = p

    chain_rev = list(reversed(chain))
    depth = len(chain_rev) - 1
    path = "/".join(str(x) for x in chain_rev)
    return depth, path


def ingest_bones_from_harvest(con, unit_id: int, model_abspath: Path) -> int:
    """
    Upsert bones (object_id keyed) from harvested <stem>_bones.json stored in harvested_json.
    """
    # Make sure blobs exist in harvested_json table (safe/idempotent)
    ingest_known_harvested_json_blobs(con, unit_id, model_abspath)

    txt = get_harvested_json_text(con, unit_id, "bones")
    if not txt:
        return 0

    try:
        bones = _parse_bones_json(txt)
    except Exception as e:
        raise HarvestJsonError(f"Failed to parse harvested bones JSON for unit_id={unit_id} ({e!r})") from e

    parent_by_obj: dict[int, int | None] = {}
    tmp_rows: list[tuple] = []

    # first pass: parent map
    for b in bones:
        try:
            obj_id = int(b.get("object_id"))
        except Exception:
            continue
        pid_raw = b.get("parent_id")
        parent_by_obj[obj_id] = int(pid_raw) if pid_raw is not None else None

    # second pass: compute depth/path + row
    for b in bones:
        try:
            name = str(b.get("name") or "")
            obj_id = int(b.get("object_id"))
            pid_raw = b.get("parent_id")
            parent_id = int(pid_raw) if pid_raw is not None else None
            pivot = b.get("pivot") or [0.0, 0.0, 0.0]
            px, py, pz = float(pivot[0]), float(pivot[1]), float(pivot[2])
        except Exception:
            continue

        depth, path = _compute_depth_and_path(parent_by_obj, obj_id)
        tmp_rows.append((unit_id, name, obj_id, parent_id, px, py, pz, depth, path))

    if not tmp_rows:
        return 0

    return upsert_bones_by_object_id(con, tmp_rows)


def _parse_boneanims_json(txt: str) -> dict[int, dict[str, Any]]:
    """
    Returns bone_object_id -> channels dict
    """
    data = json.loads(txt)
    bones = data.get("bones") or []
    out: dict[int, dict[str, Any]] = {}
    for b in bones:
        try:
            obj_id = int(b.get("object_id"))
        except Exception:
            continue
        out[obj_id] = b
    return out


def compute_motion_stats_for_unit(
    con,
    unit_id: int,
    model_abspath: Path,
    *,
    include_death_and_corpse: bool = False,
) -> KinematicsIngestResult:
    """
    Requires sequences already ingested (canonical).
    Uses harvested boneanims JSON from harvested_json table.
    """
    ACTIVE_EPS = 1e-4  # treat smaller as numeric noise
    W_RANGE = 10.0
    W_OMEGA = 2.0
    W_TRANS = 1.0
    # Ensure blobs exist (safe) and bones are ingested
    ingest_known_harvested_json_blobs(con, unit_id, model_abspath)
    bones_up = ingest_bones_from_harvest(con, unit_id, model_abspath)

    seq_rows = get_sequences_rows_for_unit(con, unit_id, include_death_and_corpse=include_death_and_corpse)
    if not seq_rows:
        return KinematicsIngestResult(bones_upserted=bones_up, motion_rows_upserted=0, seq_rows_upserted=0)

    txt = get_harvested_json_text(con, unit_id, "boneanims")
    if not txt:
        return KinematicsIngestResult(bones_upserted=bones_up, motion_rows_upserted=0, seq_rows_upserted=0)

    try:
        anims_by_obj = _parse_boneanims_json(txt)
    except Exception as e:
        raise HarvestJsonError(f"Failed to parse harvested boneanims JSON for unit_id={unit_id} ({e!r})") from e

    motion_rows: list[tuple] = []
    seq_agg_rows: list[tuple] = []

    for s in seq_rows:
        seq_id = int(s["id"])
        start_ms = int(s["start"])
        end_ms = int(s["end"])
        if end_ms < start_ms:
            continue

        active = 0
        trans_rms_list: list[float] = []
        rot_rms_list: list[float] = []

        for bone_obj, b in anims_by_obj.items():
            # Channels
            tkeys_raw = b.get("translation") or []
            rkeys_raw = b.get("rotation") or []
            skeys_raw = b.get("scaling") or []

            tkeys = []
            for k in tkeys_raw:
                try:
                    t = int(k.get("time_ms"))
                    v = k.get("value")
                    if v is None:
                        continue
                    tkeys.append(Key3(t=t, v=(float(v[0]), float(v[1]), float(v[2]))))
                except Exception:
                    continue

            rkeys = []  # <-- MUST be here, before any try/loop
            for k in rkeys_raw:
                try:
                    t = int(k.get("time_ms"))
                    q = k.get("quat") or k.get("quaternion") or k.get("value")
                    if q is None:
                        continue
                    if not isinstance(q, (list, tuple)) or len(q) != 4:
                        continue
                    rkeys.append(KeyQ(t=t, q=(float(q[0]), float(q[1]), float(q[2]), float(q[3]))))
                except Exception:
                    continue

            skeys = []
            for k in skeys_raw:
                try:
                    t = int(k.get("time_ms"))
                    v = k.get("value")
                    if v is None:
                        continue
                    skeys.append(Key3(t=t, v=(float(v[0]), float(v[1]), float(v[2]))))
                except Exception:
                    continue

            tkeys.sort(key=lambda x: x.t)
            rkeys.sort(key=lambda x: x.t)   # <-- will never UnboundLocalError now
            skeys.sort(key=lambda x: x.t)

            if rkeys_raw and not rkeys:
                print(f"[WARN] rotation keys present but none parsed for bone {bone_obj}")

            # sample times = union of in-slice key times + boundaries (deterministic)
            raw_times = []
            raw_times.extend([k.t for k in tkeys])
            raw_times.extend([k.t for k in rkeys])
            raw_times.extend([k.t for k in skeys])
            times = slice_times(raw_times, start_ms, end_ms)
            if len(times) < 2:
                # boundary-only slice => static
                times = [start_ms, end_ms]

            # reference poses at start
            p0 = sample_vec3(tkeys, start_ms)
            q0 = sample_quat(rkeys, start_ms)
            s0 = sample_vec3(skeys, start_ms)

            # mins/maxs init
            px_min = px_max = p0[0]
            py_min = py_max = p0[1]
            pz_min = pz_max = p0[2]
            smag_min = smag_max = 0.0  # translation magnitude relative to p0

            sx_min = sx_max = s0[0]
            sy_min = sy_max = s0[1]
            sz_min = sz_max = s0[2]

            rot_range = 0.0
            trans_vels: list[float] = []
            rot_omegas: list[float] = []

            prev_t = times[0]
            prev_p = sample_vec3(tkeys, prev_t)
            prev_q = sample_quat(rkeys, prev_t)

            # include first sample in ranges
            dp0 = v3_sub(prev_p, p0)
            smag = v3_len(dp0)
            smag_min = min(smag_min, smag)
            smag_max = max(smag_max, smag)
            rot_range = max(rot_range, quat_angle(q0, prev_q))

            # iterate
            for t in times[1:]:
                p = sample_vec3(tkeys, t)
                q = sample_quat(rkeys, t)
                sc = sample_vec3(skeys, t)

                # translation bounds
                px_min = min(px_min, p[0]); px_max = max(px_max, p[0])
                py_min = min(py_min, p[1]); py_max = max(py_max, p[1])
                pz_min = min(pz_min, p[2]); pz_max = max(pz_max, p[2])

                # magnitude from reference
                dp = v3_sub(p, p0)
                mag = v3_len(dp)
                smag_min = min(smag_min, mag)
                smag_max = max(smag_max, mag)

                # scale bounds
                sx_min = min(sx_min, sc[0]); sx_max = max(sx_max, sc[0])
                sy_min = min(sy_min, sc[1]); sy_max = max(sy_max, sc[1])
                sz_min = min(sz_min, sc[2]); sz_max = max(sz_max, sc[2])

                # rotation range from reference
                rot_range = max(rot_range, quat_angle(q0, q))

                # velocities
                dt_s = (t - prev_t) / 1000.0
                if dt_s > 0.0:
                    lin_v = v3_len(v3_sub(p, prev_p)) / dt_s
                    trans_vels.append(lin_v)

                    ang = quat_angle(prev_q, q)
                    rot_omegas.append(ang / dt_s)

                prev_t = t
                prev_p = p
                prev_q = q

            trans_rms = rms(trans_vels)
            rot_rms = rms(rot_omegas)
            sample_count = len(trans_vels)  # number of contributing intervals

            
            # "active" bone heuristic (deterministic, cheap):
            # active if it moves at all (either translation or rotation)
            motion_score = (
                W_RANGE * rot_range +
                W_OMEGA * rot_rms +
                W_TRANS * trans_rms
            )
            #if (smag_max > 0.0) or (rot_range > 0.0):
            if motion_score > ACTIVE_EPS:
                active += 1

            trans_rms_list.append(trans_rms)
            rot_rms_list.append(rot_rms)

            motion_rows.append(
                (
                    unit_id, seq_id, int(bone_obj),
                    px_min, px_max, py_min, py_max, pz_min, pz_max,
                    smag_min, smag_max,
                    trans_rms,
                    rot_range, rot_rms,
                    sx_min, sx_max, sy_min, sy_max, sz_min, sz_max,
                    start_ms, end_ms, sample_count,
                )
            )

        # sequence aggregates
        seq_agg_rows.append(
            (
                unit_id, seq_id,
                int(active),
                float(sum(trans_rms_list)) if trans_rms_list else 0.0,
                float(sum(rot_rms_list)) if rot_rms_list else 0.0,
            )
        )

    motion_up = upsert_bone_motion_stats(con, motion_rows) if motion_rows else 0
    seq_up = upsert_sequence_motion_stats(con, seq_agg_rows) if seq_agg_rows else 0
    # build & store sequence fingerprints (Level B)
    # Fetch unit name once (for payload readability)
    urow = con.execute("SELECT unit_name FROM units WHERE id = ?;", (unit_id,)).fetchone()
    unit_name = str(urow["unit_name"]) if urow else f"unit:{unit_id}"

    cfg = FingerprintConfig(active_eps=1e-4, w_range=10.0, w_omega=2.0, w_trans=1.0, top_k=12)

    for s in seq_rows:
        seq_id = int(s["id"])
        seq_name = str(s["name"])
        start_ms = int(s["start"])
        end_ms = int(s["end"])

        bone_rows = get_bone_stats_for_sequence(con, unit_id, seq_id)
        payload = build_sequence_fingerprint_payload(
            unit_id=unit_id,
            unit_name=unit_name,
            sequence_id=seq_id,
            sequence_name=seq_name,
            start_ms=start_ms,
            end_ms=end_ms,
            bone_rows=[dict(r) for r in bone_rows],
            cfg=cfg,
        )
        fp_json, fp_sha1 = canonicalize_fingerprint(payload)
        upsert_sequence_fingerprint(con, unit_id, seq_id, fp_json, fp_sha1)



    return KinematicsIngestResult(
        bones_upserted=bones_up,
        motion_rows_upserted=motion_up,
        seq_rows_upserted=seq_up,
    )
