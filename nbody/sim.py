# -*- coding: utf-8 -*-

import os

import numpy as np
from pyrr import Vector3
from OpenGL.GL import *
from OpenGL.GL import shaders

from .profiler import PROFILER
from . import util
from .gl_util import *


class NBodySimulation(object):
    num_galaxies = 4
    work_group_size = 256
    max_particles = work_group_size * num_galaxies * 20
    collision_overlap = 0.25
    gravity_constant = 100

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
        print('Compute buffer size: {:,d} bytes'.format(self.particles_ssbo.data.nbytes))

        self.num_particles = len(self.particles_ssbo.data)
        self.num_stars_per_galaxy = self.num_particles // self.num_galaxies
        print('Creating {:,d} galaxies with {:,d} stars each ({:,d} particles total)'.format(
            self.num_galaxies,
            self.num_stars_per_galaxy,
            self.num_particles))

        galaxy_positions = np.empty((self.num_galaxies, 3), dtype=np.float)
        for pos in galaxy_positions:
            pos[:] = util.rand_spherical(100)
        galaxy_positions = iter(galaxy_positions)

        particles = iter(self.particles_ssbo.data)
        for _ in range(self.num_galaxies):
            center_star = next(particles)
            center_star['position'] = next(galaxy_positions)
            center_star['mass'] = 1e1
            center_star['velocity'] = 0.0
            center_star['radius'] = 5.0

            for _ in range(self.num_stars_per_galaxy - 1):
                star = next(particles)

                star['mass'] = center_star['mass'] * 1e-5
                star['radius'] = 0.2

                pr, pt, ph = np.random.random((3,))
                pt *= 2 * np.pi
                ph = util.lerp(
                    ph,
                    0, 1,
                    -1, 1)
                ph *= np.exp(-pr) * center_star['radius'] / 4
                pr = util.lerp(
                    pr,
                    0, 1,
                    center_star['radius'] * 1, center_star['radius'] * 2)
                pos = Vector3([
                    pr * np.cos(pt),
                    ph,
                    pr * np.sin(pt)])

                vel = np.sqrt(self.gravity_constant * (star['mass'] + center_star['mass']) / pos.length)
                vel = vel * Vector3([-pos.z, pos.y, pos.x]).normalised

                star['position'] = center_star['position'] + pos
                star['velocity'] = vel

        glUseProgram(0)

        self.paused = True

    def update(self, dt):
        if self.paused: return

        PROFILER.begin('update')

        glUseProgram(self.shader)

        PROFILER.begin('update.uniforms')
        glUniform1ui(self.num_particles_loc, self.num_particles)
        glUniform1f(self.dt_loc, dt)

        PROFILER.begin('update.shader')
        glBindBufferBase(self.particles_ssbo.target, 0, self.particles_ssbo._buf_id)
        glDispatchCompute(self.max_particles // self.work_group_size, 1, 1)
        # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

        glUseProgram(0)

        gl_sync()

        PROFILER.end('update')
