# -*- coding: utf-8 -*-

import itertools

import numpy as np
from pyrr import Vector3

from .particle import Particle
from .profiler import PROFILER


# GRAVITY_CONSTANT = 6.674e-11
GRAVITY_CONSTANT = 100

class NBodySimulation(object):
    collision_overlap = 0.25

    def __init__(self):
        self.particles = set()

    def update(self, dt):
        for p in self.particles:
            p.acceleration = Vector3()
        PROFILER.begin('update.acc_calc')
        for p1, p2 in itertools.permutations(self.particles, 2):
            dpos = p2.position - p1.position
            dpos_len_sq = dpos.squared_length
            p1.acceleration += p2.mass / (dpos_len_sq * np.sqrt(dpos_len_sq)) * dpos
        PROFILER.begin('update.acc_apply')
        for p in self.particles:
            p.acceleration *= GRAVITY_CONSTANT
            p.velocity += p.acceleration * dt
            p.position += p.velocity * dt

        PROFILER.begin('update.collisions')
        collisions = list()
        for p1, p2 in itertools.combinations(self.particles, 2):
            if (p2.position - p1.position).squared_length <= ((p1.radius + p2.radius) * (1 - self.collision_overlap)) ** 2:
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
