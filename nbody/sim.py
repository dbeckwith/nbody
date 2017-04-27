# -*- coding: utf-8 -*-

import os
import itertools
import traceback

import numpy as np
from pyrr import Vector3
from OpenGL.GL import *
from OpenGL.GL import shaders

from .particle import Particle
from .profiler import PROFILER
from .gl_util import *


# GRAVITY_CONSTANT = 6.674e-11
GRAVITY_CONSTANT = 100

class NBodySimulation(object):
    max_particles = 256
    collision_overlap = 0.25

    def __init__(self):
        self.particles = list()

        np.random.seed(0xdeadbeef)
        for _ in range(self.max_particles):
            # https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
            r1, r2, r3 = np.random.rand(3)
            r1 *= 2 * np.pi
            r2_sqrt = 2 * np.sqrt(r2 * (1 - r2))
            r3 *= 100
            px = r3 * np.cos(r1) * r2_sqrt
            py = r3 * np.sin(r1) * r2_sqrt
            pz = r3 * (1 - 2 * r2)

            p = Particle(
                position=Vector3([px, py, pz]),
                velocity=Vector3(),
                mass=1.0,
                radius=1.0)
            self.particles.append(p)

        print('Compiling compute shader')
        try:
            with open(os.path.join(os.path.dirname(__file__), 'particle.comp'), 'r') as f:
                shader = shaders.compileShader(f.read(), GL_COMPUTE_SHADER)
            self.shader = shaders.compileProgram(shader)
        except BaseException as e:
            traceback.print_exc()
            if isinstance(e, RuntimeError):
                print(e.args[0].replace('\\n', '\n'))
            exit(1)
        glUseProgram(self.shader)

        self.num_particles_loc = glGetUniformLocation(self.shader, 'num_particles')
        self.dt_loc = glGetUniformLocation(self.shader, 'dt')

        print('Creating compute buffers')
        self.particles_ssbo = MappedBufferObject(
            target=GL_SHADER_STORAGE_BUFFER,
            dtype=np.dtype([
                ('position', np.float32, 3),
                ('mass', np.float32, 1)]),
            length=self.max_particles,
            flags=GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)
        glBindBufferBase(self.particles_ssbo.target, 0, self.particles_ssbo._buf_id)

        self.forces_ssbo = MappedBufferObject(
            target=GL_SHADER_STORAGE_BUFFER,
            dtype=np.dtype([
                ('acceleration', np.float32, 3),
                ('unused', np.float32, 1)]),
            length=self.max_particles,
            flags=GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)
        glBindBufferBase(self.forces_ssbo.target, 1, self.forces_ssbo._buf_id)

        glUseProgram(0)

    def update(self, dt):
        PROFILER.begin('update')

        glUseProgram(self.shader)

        glUniform1ui(self.num_particles_loc, len(self.particles))
        glUniform1f(self.dt_loc, dt)

        PROFILER.begin('update.copy_to_shader')
        for data, particle in zip(self.particles_ssbo.data, self.particles):
            data['position'] = particle.position
            data['mass'] = particle.mass

        PROFILER.begin('update.run_shader')
        glDispatchCompute(self.max_particles // 1, 1, 1)

        glUseProgram(0)

        gl_sync()

        PROFILER.begin('update.copy_from_shader')
        for data, particle in zip(self.forces_ssbo.data, self.particles):
            particle.acceleration = Vector3(np.copy(data['acceleration']))

        PROFILER.begin('update.acc_apply')
        for p in self.particles:
            p.acceleration *= GRAVITY_CONSTANT
            p.velocity += p.acceleration * dt
            p.position += p.velocity * dt

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
