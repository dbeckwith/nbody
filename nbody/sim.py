# -*- coding: utf-8 -*-

import itertools

import numpy as np
from pyrr import Vector3


# GRAVITY_CONSTANT = 6.674e-11
GRAVITY_CONSTANT = 1.0

class NBodySimulation(object):
    def __init__(self):
        self.particles = []

    def update(self, dt):
        for p in self.particles:
            p.position.z -= dt / 10
        return

        for p in self.particles:
            p.acceleration = Vector3()
        for p1, p2 in itertools.permutations(self.particles, 2):
            p1.acceleration += GRAVITY_CONSTANT * p2.mass / (p2.position - p1.position).squared_length
        for p in self.particles:
            p.velocity += p.acceleration * dt
            p.position += p.velocity * dt
