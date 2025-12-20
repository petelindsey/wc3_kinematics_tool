#motion_features.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x, y, z, w)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def v3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v3_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v3_len(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def q_dot(a: Quat, b: Quat) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]


def q_norm(q: Quat) -> Quat:
    n = math.sqrt(q_dot(q, q))
    if n <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / n
    return (q[0]*inv, q[1]*inv, q[2]*inv, q[3]*inv)


def q_slerp(a: Quat, b: Quat, t: float) -> Quat:
    """
    Deterministic slerp with sign correction for shortest arc.
    """
    a = q_norm(a)
    b = q_norm(b)
    d = q_dot(a, b)
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d

    d = _clamp(d, 0.0, 1.0)

    if d > 0.9995:
        # Nearly identical: nlerp
        out = (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t, a[3] + (b[3]-a[3])*t)
        return q_norm(out)

    theta = math.acos(d)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    out = (a[0]*w1 + b[0]*w2, a[1]*w1 + b[1]*w2, a[2]*w1 + b[2]*w2, a[3]*w1 + b[3]*w2)
    return q_norm(out)


def quat_angle(a: Quat, b: Quat) -> float:
    """
    Angle between quaternions in radians, sign-invariant.
    """
    a = q_norm(a)
    b = q_norm(b)
    d = abs(q_dot(a, b))
    d = _clamp(d, 0.0, 1.0)
    return 2.0 * math.acos(d)


@dataclass(frozen=True)
class Key3:
    t: int
    v: Vec3


@dataclass(frozen=True)
class KeyQ:
    t: int
    q: Quat


def _find_bracketing_keys(keys: list, t: int) -> tuple[Optional[object], Optional[object]]:
    """
    Returns (prev, next) where prev.t <= t <= next.t if possible.
    Assumes keys sorted by t.
    """
    if not keys:
        return (None, None)
    if t <= keys[0].t:
        return (keys[0], keys[0])
    if t >= keys[-1].t:
        return (keys[-1], keys[-1])

    lo = 0
    hi = len(keys) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keys[mid].t <= t:
            lo = mid
        else:
            hi = mid
    return (keys[lo], keys[hi])


def sample_vec3(keys: list[Key3], t: int) -> Vec3:
    if not keys:
        return (0.0, 0.0, 0.0)
    a, b = _find_bracketing_keys(keys, t)
    assert a is not None and b is not None
    if a.t == b.t:
        return a.v
    alpha = (t - a.t) / float(b.t - a.t)
    return v3_lerp(a.v, b.v, alpha)


def sample_quat(keys: list[KeyQ], t: int) -> Quat:
    if not keys:
        return (0.0, 0.0, 0.0, 1.0)
    a, b = _find_bracketing_keys(keys, t)
    assert a is not None and b is not None
    if a.t == b.t:
        return q_norm(a.q)
    alpha = (t - a.t) / float(b.t - a.t)
    return q_slerp(a.q, b.q, alpha)


def slice_times(raw_times: Iterable[int], start_ms: int, end_ms: int) -> list[int]:
    ts = {int(start_ms), int(end_ms)}
    for t in raw_times:
        if start_ms <= int(t) <= end_ms:
            ts.add(int(t))
    out = sorted(ts)
    return out


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    s = 0.0
    for v in values:
        s += v * v
    return math.sqrt(s / float(len(values)))
