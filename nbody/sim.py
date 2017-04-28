# -*- coding: utf-8 -*-

import os
import itertools
import copy

import numpy as np
from pyrr import Vector3
from OpenGL.GL import *
from OpenGL.GL import shaders

from .profiler import PROFILER
from .gl_util import *


class NBodySimulation(object):
    max_particles = 256
    collision_overlap = 0.25

    def __init__(self):
        print('Compiling compute shader')
        with open(os.path.join(os.path.dirname(__file__), 'particle.comp'), 'r') as f:
            shader = shaders.compileShader(f.read(), GL_COMPUTE_SHADER)
        self.shader = shaders.compileProgram(shader)
        glUseProgram(self.shader)

        self.num_particles_loc = glGetUniformLocation(self.shader, 'num_particles')
        self.dt_loc = glGetUniformLocation(self.shader, 'dt')

        print('Creating compute buffer')
        self.particles_ssbo = MappedBufferObject(
            target=GL_SHADER_STORAGE_BUFFER,
            dtype=np.dtype([
                ('position', np.float32, 3),
                ('mass', np.float32, 1),
                ('velocity', np.float32, 3),
                ('radius', np.float32, 1)]),
            length=self.max_particles,
            flags=GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)

        print('Creating particles')
        np.random.seed(0xdeadbeef)
        self.num_particles = len(self.particles_ssbo.data)
        for particle in self.particles_ssbo.data:
            # https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
            r1, r2, r3 = np.random.rand(3)
            r1 *= 2 * np.pi
            r2_sqrt = 2 * np.sqrt(r2 * (1 - r2))
            r3 *= 100
            px = r3 * np.cos(r1) * r2_sqrt
            py = r3 * np.sin(r1) * r2_sqrt
            pz = r3 * (1 - 2 * r2)

            particle['position'] = [px, py, pz]
            particle['mass'] = 1.0
            particle['velocity'] = [0.0, 0.0, 0.0]
            particle['radius'] = 1.0

        glUseProgram(0)

    def update(self, dt):
        PROFILER.begin('update')

        glUseProgram(self.shader)

        PROFILER.begin('update.uniforms')
        glUniform1ui(self.num_particles_loc, self.num_particles)
        glUniform1f(self.dt_loc, dt)

        PROFILER.begin('update.shader')
        glBindBufferBase(self.particles_ssbo.target, 0, self.particles_ssbo._buf_id)
        glDispatchCompute(self.max_particles // 256, 1, 1)
        # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

        glUseProgram(0)

        gl_sync()

        PROFILER.end('update')

        return


        PROFILER.begin('update')

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

        PROFILER.begin('update.collisions.group')
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

        PROFILER.begin('update.collisions.combine')
        for group in collisions:
            self.particles -= group
            self.particles.add(Particle.sum(group))

        PROFILER.end('update')
