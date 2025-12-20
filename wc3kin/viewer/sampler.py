from __future__ import annotations

import math
from typing import Iterable

from .types import KeyQuat, KeyVec3, Quat, Vec3


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp_vec3(a: Vec3, b: Vec3, t: float) -> Vec3:
    t = _clamp(t, 0.0, 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def quat_dot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_normalize(q: Quat) -> Quat:
    n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if n2 <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / math.sqrt(n2)
    return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


def quat_slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    """
    Deterministic shortest-arc slerp.
    Assumes quats are (x,y,z,w).
    """
    t = _clamp(t, 0.0, 1.0)
    q0n = quat_normalize(q0)
    q1n = quat_normalize(q1)

    d = quat_dot(q0n, q1n)
    # shortest path
    if d < 0.0:
        q1n = (-q1n[0], -q1n[1], -q1n[2], -q1n[3])
        d = -d

    # If very close, lerp + normalize to avoid numerical issues
    if d > 0.9995:
        out = (
            q0n[0] + (q1n[0] - q0n[0]) * t,
            q0n[1] + (q1n[1] - q0n[1]) * t,
            q0n[2] + (q1n[2] - q0n[2]) * t,
            q0n[3] + (q1n[3] - q0n[3]) * t,
        )
        return quat_normalize(out)

    theta_0 = math.acos(_clamp(d, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-12:
        return q0n

    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - d * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    out = (
        q0n[0] * s0 + q1n[0] * s1,
        q0n[1] * s0 + q1n[1] * s1,
        q0n[2] * s0 + q1n[2] * s1,
        q0n[3] * s0 + q1n[3] * s1,
    )
    return quat_normalize(out)


def _find_bracketing_keys(times: list[int], t_ms: int) -> tuple[int, int]:
    """
    Returns (i0, i1) indices into times such that times[i0] <= t <= times[i1].
    If t is outside range, returns nearest endpoint pair (0,0) or (n-1,n-1).
    """
    n = len(times)
    if n == 0:
        return (0, 0)
    if t_ms <= times[0]:
        return (0, 0)
    if t_ms >= times[-1]:
        return (n - 1, n - 1)

    # binary search
    lo, hi = 0, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if times[mid] <= t_ms:
            lo = mid
        else:
            hi = mid
    return (lo, hi)


def sample_vec3(keys: list[KeyVec3], t_ms: int, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if not keys:
        return default
    if len(keys) == 1:
        return keys[0].value

    times = [k.time_ms for k in keys]
    i0, i1 = _find_bracketing_keys(times, t_ms)
    if i0 == i1:
        return keys[i0].value

    k0, k1 = keys[i0], keys[i1]
    dt = float(k1.time_ms - k0.time_ms)
    if dt <= 0.0:
        return k0.value
    alpha = float(t_ms - k0.time_ms) / dt
    return lerp_vec3(k0.value, k1.value, alpha)


def sample_quat(keys: list[KeyQuat], t_ms: int, default: Quat = (0.0, 0.0, 0.0, 1.0)) -> Quat:
    if not keys:
        return default
    if len(keys) == 1:
        return quat_normalize(keys[0].quat)

    times = [k.time_ms for k in keys]
    i0, i1 = _find_bracketing_keys(times, t_ms)
    if i0 == i1:
        return quat_normalize(keys[i0].quat)

    k0, k1 = keys[i0], keys[i1]
    dt = float(k1.time_ms - k0.time_ms)
    if dt <= 0.0:
        return quat_normalize(k0.quat)
    alpha = float(t_ms - k0.time_ms) / dt
    return quat_slerp(k0.quat, k1.quat, alpha)
