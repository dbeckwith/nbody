# -*- coding: utf-8 -*-

import itertools

import numpy as np
from pyrr import Vector3


# GRAVITY_CONSTANT = 6.674e-11
GRAVITY_CONSTANT = 0.01

class NBodySimulation(object):
    def __init__(self):
        self.particles = []

    def update(self, dt):
        for p in self.particles:
            p.acceleration = Vector3()
        for p1, p2 in itertools.permutations(self.particles, 2):
            dpos = p2.position - p1.position
            dpos_len_sq = dpos.squared_length
            p1.acceleration += GRAVITY_CONSTANT * p2.mass / (dpos_len_sq * np.sqrt(dpos_len_sq)) * dpos
        for p in self.particles:
            p.velocity += p.acceleration * dt
            p.position += p.velocity * dt
