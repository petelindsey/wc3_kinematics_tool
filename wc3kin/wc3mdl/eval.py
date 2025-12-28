# wc3kin/wc3mdl/eval.py
"""
Deterministic animation evaluation for Warcraft 3 models.
"""

import numpy as np
from typing import Dict

def quat_to_mat(q):
    x,y,z,w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w,   0],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w,   0],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y, 0],
        [0,0,0,1]
    ])

def translate(v):
    m = np.eye(4)
    m[:3,3] = v
    return m

def scale(v):
    return np.diag([v[0], v[1], v[2], 1])

def evaluate_pose(model, clip, t_ms: int) -> Dict[int, np.ndarray]:
    world = {}

    for bone_id, node in model.nodes.items():
        anim = clip.bone_anims.get(bone_id)

        T = translate(anim.translation.get(t_ms, (0,0,0)))
        R = quat_to_mat(anim.rotation.get(t_ms, (0,0,0,1)))
        S = scale(anim.scale.get(t_ms, (1,1,1)))
        P = translate(node.pivot)
        Pinv = translate((-node.pivot[0], -node.pivot[1], -node.pivot[2]))

        local = P @ T @ R @ S @ Pinv

        if node.parent is None:
            world[bone_id] = local
        else:
            world[bone_id] = world[node.parent] @ local

    return world

def build_skin_matrices(world_mats, inv_bind_world):
    return {
        i: world_mats[i] @ inv_bind_world[i]
        for i in world_mats
    }
