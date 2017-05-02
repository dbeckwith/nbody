# -*- coding: utf-8 -*-

import numpy as np
from pyrr import Vector3


def rand_spherical(r):
    """
    Creates a random vector uniformly distributed on a sphere of given radius.

    Arguments:
        r: float, the radius of the sphere

    Returns:
        Vector3
    """
    r1, r2, r3 = np.random.random(3)
    r1 *= 2 * np.pi
    r2_sqrt = 2 * np.sqrt(r2 * (1 - r2))
    r3 *= r
    x = r3 * np.cos(r1) * r2_sqrt
    y = r3 * np.sin(r1) * r2_sqrt
    z = r3 * (1 - 2 * r2)
    return Vector3([x, y, z])

def from_spherical(r, t, p):
    """
    Creates a vector in Cartesian coordinates in the same location as the given spherical coordinates.

    Arguments:
        r: float, the radius of the point
        t: float, the theta angle of the point in radians
        p: float, the phi angle of the point in radians

    Returns:
        Vector3
    """
    if not r: return Vector3()
    sin_p = np.sin(p)
    x = r * np.cos(t) * sin_p
    y = r * np.sin(t) * sin_p
    z = r * np.cos(p)
    return Vector3([x, y, z])

def to_spherical(v):
    """
    Determines the spherical coordinates of a vector in Cartesian coordinates.

    Arguments:
        v: Vector3, the position in Cartesian coordinates

    Returns:
        r: float, the radius of the point
        t: float, the theta angle of the point in radians
        p: float, the phi angle of the point in radians
    """
    if not v.any(): return 0, 0, 0
    r = v.length
    t = np.arctan2(v.y, v.x)
    p = np.arccos(v.z / r)
    return r, t, p

def lerp(x, old_min, old_max, new_min, new_max):
    """
    Linearly interpolates between the ranges given.

    Arguments:
        x: number, the value to interpolate
        old_min: number, the minimum to interpolate from
        old_max: number, the maximum to interpolate from
        new_min: number, the minimum to interpolate to
        new_max: number, the maximum to interpolate to

    Returns:
        number, the interpolated value
    """
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
