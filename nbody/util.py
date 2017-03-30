# -*- coding: utf-8 -*-

import numpy as np
from pyrr import Vector3


def from_spherical(r, t, p):
    if not r: return Vector3()
    sin_p = np.sin(p)
    x = r * np.cos(t) * sin_p
    y = r * np.sin(t) * sin_p
    z = r * np.cos(p)
    return Vector3([x, y, z])

def to_spherical(v):
    if not v.any(): return 0, 0, 0
    r = v.length
    t = np.arctan2(v.y, v.x)
    p = np.arccos(v.z / r)
    return r, t, p

def lerp(x, old_min, old_max, new_min, new_max):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
