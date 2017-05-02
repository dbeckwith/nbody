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
    """
    Class for handling the simulation of the N-body problem.
    """

    num_galaxies = 4 # number of galaxies to generate
    work_group_size = 256 # compute shader work group size
    max_particles = work_group_size * num_galaxies * 20 # total starting particles
    collision_overlap = 0.25 # max overlap allowed between particles before they collide
    gravity_constant = 100.0
    starting_area_radius = 100.0 # maximum distance from origin galaxies can spawn
    center_star_mass = 1.0e1 # mass of center star of galaxies
    center_star_radius = 5.0 # radius of center star of galaxies
    galaxy_star_mass_factor = 1.0e-5 # mass factor for oribiting stars

    def __init__(self):
        print('Compiling compute shader')
        with open(os.path.join(os.path.dirname(__file__), 'particle.comp'), 'r') as f:
            shader = shaders.compileShader(f.read(), GL_COMPUTE_SHADER)
        self.shader = shaders.compileProgram(shader)
        glUseProgram(self.shader)

        # assign uniform constant
        glUniform1f(glGetUniformLocation(self.shader, 'gravity_constant'), self.gravity_constant)
        # save variable uniform locations
        self.num_particles_loc = glGetUniformLocation(self.shader, 'num_particles')
        self.dt_loc = glGetUniformLocation(self.shader, 'dt')

        print('Creating compute buffer')
        # create persistant memory-mapped buffer to share memory with GPU and allow fast transfer
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
        # divide total particles amoung galaxies
        self.num_stars_per_galaxy = self.num_particles // self.num_galaxies
        print('Creating {:,d} galaxies with {:,d} stars each ({:,d} particles total)'.format(
            self.num_galaxies,
            self.num_stars_per_galaxy,
            self.num_particles))

        # determine galaxy positions first
        # this way, if the star creation is altered, the same random seed will still give the same galaxy positions
        galaxy_positions = np.empty((self.num_galaxies, 3), dtype=np.float)
        for pos in galaxy_positions:
            pos[:] = util.rand_spherical(self.starting_area_radius)
        galaxy_positions = iter(galaxy_positions)

        # generate particles
        particles = iter(self.particles_ssbo.data)
        for _ in range(self.num_galaxies):
            center_star = next(particles)
            center_star['position'] = next(galaxy_positions)
            center_star['mass'] = self.center_star_mass
            center_star['velocity'] = 0.0
            center_star['radius'] = self.center_star_radius

            # center star is included in count, so num_stars_per_galaxy - 1
            for _ in range(self.num_stars_per_galaxy - 1):
                star = next(particles)

                # mass is center * mass factor
                star['mass'] = center_star['mass'] * self.galaxy_star_mass_factor
                # radius is scaled so orbiting stars have same density as center stars
                star['radius'] = np.cbrt(star['mass'] / center_star['mass']) * center_star['radius']

                # determine star positions
                # we want them to be randomly distributed in a disk around the center star
                # there should also be a random height perturbation from the disk
                # which is based on the radius from the center

                # position theta
                pt = np.random.uniform(0, 2 * np.pi)

                # position radius, higher chance of being closer
                pr = np.random.lognormal(sigma=0.5)
                # scale by center star radius
                pr = util.lerp(
                    pr,
                    0, 1,
                    center_star['radius'] * 0.5, center_star['radius'] * 1.0)

                # position height, first unscale radius
                ph = pr / center_star['radius']
                # height envelope increases exponentially as radius gets smaller
                ph = np.exp(-ph) * center_star['radius'] / 3
                # right now ph is the max/min height, so pick random in that range
                ph = np.random.uniform(-ph, ph)

                # make position vector
                pos = Vector3([
                    pr * np.cos(pt),
                    ph,
                    pr * np.sin(pt)])

                # calculate velocity needed to stay in orbit
                vel = np.sqrt(self.gravity_constant * (star['mass'] + center_star['mass']) / pos.length)
                # velocity direction is perpendicular to position
                vel = vel * Vector3([-pos.z, pos.y, pos.x]).normalised

                star['position'] = center_star['position'] + pos
                star['velocity'] = vel

        glUseProgram(0)

    def update(self, dt):
        """
        Updates a single frame of the simulation.

        Arguments:
            dt: float in (0, inf), time in seconds since last update
        """
        PROFILER.begin('update')

        glUseProgram(self.shader)

        PROFILER.begin('update.uniforms')
        # update variable uniforms, number of particles and timestep
        glUniform1ui(self.num_particles_loc, self.num_particles)
        glUniform1f(self.dt_loc, dt)

        PROFILER.begin('update.shader')
        # bind particle data buffer to shader buffer 0
        glBindBufferBase(self.particles_ssbo.target, 0, self.particles_ssbo._buf_id)
        # dispatch compute shader with max_particles / work_group_size workers
        # compute shader will calculate gravity forces and update particle data
        glDispatchCompute(self.max_particles // self.work_group_size, 1, 1)
        # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        # glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

        glUseProgram(0)

        # wait for compute shader to finish
        gl_sync()

        PROFILER.end('update')
