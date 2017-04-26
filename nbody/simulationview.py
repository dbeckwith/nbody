# -*- coding: utf-8 -*-

import os
import time
import traceback

import numpy as np
from pyrr import Vector3, Vector4, Matrix44

import OpenGL
OpenGL.ERROR_CHECKING = True
OpenGL.FULL_LOGGING = True
from OpenGL.GL import *
from OpenGL.GL import shaders

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import QOpenGLWidget

from .sim import NBodySimulation
from .profiler import PROFILER
from . import util
from .gl_util import *


class SimulationView(QOpenGLWidget):
    fps = 60
    profiler_print_interval = 1.0

    def __init__(self, parent):
        super().__init__(parent)

        self.size = QSize(800, 500)

        self.camera = OrbitCamera(
            distance=100.0,
            azimuth=0.0,
            zenith=np.pi / 2,
            fovx=np.deg2rad(90.0),
            aspect=self.size.width() / self.size.height(),
            near=0.01,
            far=1000.0)

    def sizeHint(self):
        return self.size

    def initializeGL(self):
        print('Initializing...')

        print_gl_version()

        print('Initializing sim')
        self.sim = NBodySimulation()

        print('Setting OpenGL options')
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        print('Compiling render shaders')
        try:
            with open(os.path.join(os.path.dirname(__file__), 'particle.vert'), 'r') as f:
                vshader = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
            with open(os.path.join(os.path.dirname(__file__), 'particle.frag'), 'r') as f:
                fshader = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
            self.shader = shaders.compileProgram(vshader, fshader)
        except BaseException as e:
            traceback.print_exc()
            if isinstance(e, RuntimeError):
                print(e.args[0].replace('\\n', '\n'))
            exit(1)
        glUseProgram(self.shader)

        print('Mapping uniforms')
        self.view_loc = glGetUniformLocation(self.shader, 'view')
        self.proj_loc = glGetUniformLocation(self.shader, 'proj')

        glUniform1f(glGetUniformLocation(self.shader, 'collision_overlap'), self.sim.collision_overlap)
        glUniform1ui(glGetUniformLocation(self.shader, 'color_mode'), 1)

        print('Generating VAO')
        self.particles_vao = glGenVertexArrays(1)
        glBindVertexArray(self.particles_vao)

        print('Generating sprite data buffer')
        self.sprite_data_vbo = ConstBufferObject(
            usage=GL_STATIC_DRAW,
            target=GL_ARRAY_BUFFER,
            data=np.array(
                [([-1.0,  1.0], [0.0, 1.0]),
                 ([-1.0, -1.0], [0.0, 0.0]),
                 ([ 1.0,  1.0], [1.0, 1.0]),
                 ([ 1.0, -1.0], [1.0, 0.0])],
                dtype=np.dtype([
                    ('vertex', np.float32, 2),
                    ('uv', np.float32, 2)])))
        setup_vbo_attrs(
            vbo=self.sprite_data_vbo,
            shader=self.shader,
            attr_prefix='sprite_')

        print('Generating particle data buffer')
        self.particle_data_vbo = MappedBufferObject(
            target=GL_ARRAY_BUFFER,
            dtype=np.dtype([
                ('radius', np.float32),
                ('mass', np.float32),
                ('position', np.float32, 3),
                ('velocity', np.float32, 3)]),
            length=self.sim.max_particles,
            flags=GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)
        setup_vbo_attrs(
            vbo=self.particle_data_vbo,        
            shader=self.shader,
            attr_prefix='particle_',
            divisor=1)

        glBindVertexArray(0)
        glUseProgram(0)

        print('Starting animation timer')
        self.last_update_time = time.time() - 1 / self.fps
        self.ani_timer = QTimer(self)
        self.ani_timer.setInterval(1000 / self.fps)
        self.ani_timer.timeout.connect(self.update)
        self.ani_timer.start()

        self.last_profiler_print_time = time.time()

        print('Done initializing')

    def update(self):
        PROFILER.begin()

        t = time.time()
        dt = t - self.last_update_time
        self.last_update_time = t

        self.sim.update(dt)

        super().update()

        PROFILER.end()

        if t - self.last_profiler_print_time >= self.profiler_print_interval:
            PROFILER.print_stages()
            PROFILER.reset()
            self.last_profiler_print_time = t

    def paintGL(self):
        PROFILER.begin('render')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # TODO: try to get multisampling working
        # glEnable(GL_MULTISAMPLE)
        # print(glGetIntegerv(GL_SAMPLES), glGetIntegerv(GL_SAMPLE_BUFFERS))

        glUseProgram(self.shader)
        glBindVertexArray(self.particles_vao)

        PROFILER.begin('render.camera')
        self.camera.update()
        glUniformMatrix4fv(self.view_loc, 1, GL_TRUE, self.camera.view.astype(np.float32))
        glUniformMatrix4fv(self.proj_loc, 1, GL_TRUE, self.camera.proj.astype(np.float32))
        PROFILER.end('render.camera')

        PROFILER.begin('render.particles.data')
        # TODO: maybe don't need to sort ever? maybe only don't need if no transparency?
        for data, particle in zip(self.particle_data_vbo.data, sorted(self.sim.particles, key=self._particle_sort, reverse=True)):
            data['radius'] = particle.radius
            data['mass'] = particle.mass
            data['position'] = particle.position
            data['velocity'] = particle.velocity
        PROFILER.end('render.particles.data')

        PROFILER.begin('render.particles.draw')
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, self.sprite_data_vbo.length, len(self.sim.particles))
        gl_sync()
        PROFILER.end('render.particles.draw')

        glBindVertexArray(0)
        glUseProgram(0)

        PROFILER.end('render')

    def resizeGL(self, width, height):
        self.size = QSize(width, height)
        self.camera.aspect = self.size.width() / self.size.height()

    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.drag_mouse_start = event.pos()
            self.drag_cam_start = (self.camera.azimuth, self.camera.zenith)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton == Qt.LeftButton:
            drag = event.pos() - self.drag_mouse_start
            drag = np.array([drag.x(), -drag.y()], dtype=np.float)
            drag = util.lerp(drag, 0, 500, 0, np.pi)
            azimuth, zenith = self.drag_cam_start
            azimuth += drag[0]
            zenith += drag[1]
            self.camera.azimuth = azimuth
            self.camera.zenith = zenith

    def wheelEvent(self, event):
        scroll = event.angleDelta().y()
        zoom = util.lerp(-scroll, 0, 120, 0, 0.25)
        self.camera.distance = 2 ** (np.log2(self.camera.distance) + zoom)
        self.camera.near = 2 ** (np.log2(self.camera.near) + zoom)
        self.camera.far = 2 ** (np.log2(self.camera.far) + zoom)

    def _particle_sort(self, particle):
        return (self.camera.eye - particle.position).squared_length

class Camera(object):
    def __init__(self, eye, at, up, fovx, aspect, near, far):
        self.eye = eye
        self.at = at
        self.up = up
        self.fovx = fovx
        self.aspect = aspect
        self.near = near
        self.far = far

        self.update()

    def update(self):
        eye = self.eye
        at = self.at
        up = self.up

        n = (eye - at).normalised
        u = up.cross(n).normalised
        v = n.cross(u)
        self.view = Matrix44(
            [[u.x, u.y, u.z, -u.dot(eye)],
             [v.x, v.y, v.z, -v.dot(eye)],
             [n.x, n.y, n.z, -n.dot(eye)],
             [  0,   0,   0,           1]])

        fovx = self.fovx
        aspect = self.aspect
        near = self.near
        far = self.far

        recip_tan_fov = 1 / np.tan(fovx / 2)
        a = recip_tan_fov
        b = aspect * recip_tan_fov
        c = -(far + near) / (far - near)
        d = -2 * far * near / (far - near)
        self.proj = Matrix44(
            [[a, 0,  0, 0],
             [0, b,  0, 0],
             [0, 0,  c, d],
             [0, 0, -1, 0]])

class OrbitCamera(Camera):
    zenith_eps = 1e-5 * np.pi

    def __init__(self, distance, azimuth, zenith, fovx, aspect, near, far):
        self.distance = distance
        self.azimuth = azimuth
        self.zenith = zenith
        super().__init__(
            Vector3(),
            Vector3(),
            Vector3([0.0, 1.0, 0.0]),
            fovx,
            aspect,
            near,
            far)

    def update(self):
        self.azimuth %= np.pi * 2
        self.zenith = np.clip(self.zenith, self.zenith_eps, np.pi - self.zenith_eps)
        eye = util.from_spherical(self.distance, self.azimuth, self.zenith)
        self.eye = Vector3([eye.x, eye.z, eye.y])
        super().update()
