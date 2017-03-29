# -*- coding: utf-8 -*-

import itertools

import numpy as np
from pyrr import Vector3

from .particle import Particle


# GRAVITY_CONSTANT = 6.674e-11
GRAVITY_CONSTANT = 0.1

class NBodySimulation(object):
    collision_overlap = 0.5

    def __init__(self):
        self.particles = set()

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

        collisions = list()
        for p1, p2 in itertools.combinations(self.particles, 2):
            if (p2.position - p1.position).squared_length <= (p1.radius + p2.radius) ** 2 * (1 - self.collision_overlap):
                found_group = False
                for group in collisions:
                    if p1 in group:
                        group.add(p2)
                        found_group = True
                    elif p2 in group:
                        group.add(p1)
                        found_group = True
                    if found_group:
                        break
                if not found_group:
                    collisions.append({p1, p2})
        for group in collisions:
            self.particles -= group
            self.particles.add(Particle.sum(group))
