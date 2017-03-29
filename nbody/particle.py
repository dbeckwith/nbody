# -*- coding: utf-8 -*-

import numpy as np
from pyrr import Vector3


class Particle(object):
    def __init__(self, position=None, velocity=None, mass=None, radius=None):
        self.position = Vector3() if position is None else position
        self.velocity = Vector3() if velocity is None else velocity
        self.mass = 1 if mass is None else mass
        self.radius = 1 if radius is None else radius

    @property
    def momentum(self):
        return self.mass * self.velocity

    @momentum.setter
    def momentum(self, momentum):
        self.velocity = momentum / self.mass

    @property
    def volume(self):
        return 4 / 3 * np.pi * self.radius * self.radius * self.radius

    @volume.setter
    def volume(self, volume):
        self.radius = np.cbrt(volume * 3 / 4 / np.pi)

    @property
    def density(self):
        return self.mass / self.volume

    @density.setter
    def density(self, density):
        self.volume = density / self.mass

    def copy(self):
        return Particle(
            self.position,
            self.velocity,
            self.mass,
            self.radius)

    @staticmethod
    def sum(particles):
        mass = 0
        momentum = Vector3()
        volume = 0
        for p in particles:
            mass += p.mass
            volume += p.volume
            momentum += p.momentum
        momentum /= len(particles)
        
        s = Particle()
        s.mass = mass
        s.momentum = momentum
        s.volume = volume
        s.position = sum(p.mass * p.position for p in particles) / mass
        return s

    def __repr__(self):
        return 'Particle(' + \
            'position=' + repr(self.position) + ', ' + \
            'velocity=' + repr(self.velocity) + ', ' + \
            'mass=' + repr(self.mass) + ', ' + \
            'radius=' + repr(self.radius) + ')'
