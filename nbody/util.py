# -*- coding: utf-8 -*-

import time

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

class UpdateTimer(object):
    def __init__(self, print_interval=None, print_fn=None, max_samples=1000):
        self.print_interval = print_interval
        self.print_fn = print_fn
        if self.print_fn is not None:
            self.print_timer = time.time()
        else:
            self.print_timer = None
        self.max_samples = max_samples
        self.ups_samples = np.empty((self.max_samples,), dtype=np.float32)
        self.num_samples = 0
        self.sample_pos = 0
        self.timer = None

    def update(self):
        t = time.time()
        if self.timer is None:
            self.timer = t
            return
        dt = t - self.timer
        self.timer = t
        ups = 1 / dt

        self.num_samples += 1
        self.num_samples = min(self.num_samples, self.max_samples)

        self.ups_samples[self.sample_pos] = ups

        self.sample_pos += 1
        self.sample_pos %= self.max_samples

        if self.print_timer is not None and t - self.print_timer >= self.print_interval:
            self.print_timer = t
            self.print_fn(self.capture_ups())

    def capture_ups(self):
        if self.num_samples == self.max_samples:
            ups = np.mean(self.ups_samples)
        else:
            ups = np.mean(self.ups_samples[:self.num_samples])
        self.num_samples = 0
        self.sample_pos = 0
        return ups
