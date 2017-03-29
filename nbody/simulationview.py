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
from OpenGL.arrays.vbo import VBO
from ctypes import c_void_p

from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtWidgets import QOpenGLWidget

from .sim import NBodySimulation


class SimulationView(QOpenGLWidget):
    max_particles = 256

    def __init__(self, parent):
        super().__init__(parent)

        self.size = QSize(800, 500)

        self.camera = Camera(
            eye=Vector3([0.0, 0.0, 10.0]),
            at=Vector3([0.0, 0.0, 0.0]),
            up=Vector3([0.0, 1.0, 0.0]),
            fovx=np.deg2rad(90.0),
            aspect=self.size.width() / self.size.height(),
            near=0.01,
            far=100.0)

        self.sim = NBodySimulation()

    def sizeHint(self):
        return self.size

    def initializeGL(self):
        print_gl_version()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        try:
            with open(os.path.join(os.path.dirname(__file__), 'sim.vert'), 'r') as f:
                vshader = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
            with open(os.path.join(os.path.dirname(__file__), 'sim.frag'), 'r') as f:
                fshader = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
            self.shader = shaders.compileProgram(vshader, fshader)
        except BaseException as e:
            traceback.print_exc()
            if isinstance(e, RuntimeError):
                print(e.args[0].replace('\\n', '\n'))
            exit(1)
        glUseProgram(self.shader)

        self.view_loc = glGetUniformLocation(self.shader, 'view')
        self.proj_loc = glGetUniformLocation(self.shader, 'proj')

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.sprite_data = np.array(
            [([-1.0,  1.0], [0.0, 1.0]),
             ([-1.0, -1.0], [0.0, 0.0]),
             ([ 1.0,  1.0], [1.0, 1.0]),
             ([ 1.0, -1.0], [1.0, 0.0])],
            dtype=np.dtype([
                ('vertex', np.float32, 2),
                ('uv', np.float32, 2)]))
        self.sprite_data_vbo = make_vbo(
            data=self.sprite_data,
            usage='GL_STATIC_DRAW',
            target='GL_ARRAY_BUFFER',
            shader=self.shader,
            attr_prefix='sprite_')

        sprite_img = self._gen_sprite_img()
        self.sprite_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.sprite_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            sprite_img.shape[0],
            sprite_img.shape[1],
            0,
            GL_RED,
            GL_FLOAT,
            sprite_img)
        glUniform1i(glGetUniformLocation(self.shader, 'tex'), 0)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.particle_data = np.empty(
            (self.max_particles,),
            dtype=np.dtype([
                ('radius', np.float32),
                ('position', np.float32, 3)]))
        self.particle_data_vbo = make_vbo(
            data=self.particle_data,
            usage='GL_STREAM_DRAW',
            target='GL_ARRAY_BUFFER',
            shader=self.shader,
            attr_prefix='particle_',
            divisor=1)

        glBindVertexArray(0)
        glUseProgram(0)

        fps = 60
        self.last_update_time = time.time() - 1 / fps
        self.ani_timer = QTimer(self)
        self.ani_timer.setInterval(1000 / fps)
        self.ani_timer.timeout.connect(self.update)
        self.ani_timer.start()

    def update(self):
        t = time.time()
        dt = t - self.last_update_time

        self.sim.update(dt)

        super().update()

        self.last_update_time = t

    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        for i, particle in enumerate(sorted(self.sim.particles, key=self._particle_sort, reverse=True)):
            d = self.particle_data[i]
            d['radius'] = particle.radius
            d['position'] = particle.position

        with self.particle_data_vbo:
            self.particle_data_vbo.set_array(self.particle_data[:len(self.sim.particles)])

        glBindTexture(GL_TEXTURE_2D, self.sprite_texture)

        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, len(self.sprite_data), len(self.sim.particles))

        glBindTexture(GL_TEXTURE_2D, 0)

        glBindVertexArray(0)
        glUseProgram(0)

    def resizeGL(self, width, height):
        self.size = QSize(width, height)
        self.camera.aspect = self.size.width() / self.size.height()
        self.camera_update()

    def camera_update(self):
        self.camera.update()

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.view_loc, 1, GL_TRUE, self.camera.view.astype(np.float32))
        glUniformMatrix4fv(self.proj_loc, 1, GL_TRUE, self.camera.proj.astype(np.float32))

        glBindVertexArray(0)
        glUseProgram(0)

    def _particle_sort(self, particle):
        return (self.camera.eye - particle.position).squared_length

    def _gen_sprite_img(self):
        resolution = 64
        ramp_pwr = -1.5

        img = np.zeros((resolution, resolution), dtype=np.complex)
        l = np.linspace(-1, 1, resolution)
        img.real, img.imag = np.meshgrid(l, l)
        img = np.abs(img) ** (2 ** ramp_pwr)
        img = 1 - img
        return img.astype(np.float32)

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

        inv_tan_fov = 1 / np.tan(fovx / 2)
        a = inv_tan_fov
        b = aspect * inv_tan_fov
        c = -(far + near) / (far - near)
        d = -2 * far * near / (far - near)
        self.proj = Matrix44(
            [[a, 0,  0, 0],
             [0, b,  0, 0],
             [0, 0,  c, d],
             [0, 0, -1, 0]])

def print_gl_version():
        version_str = str(glGetString(GL_VERSION), 'utf-8')
        shader_version_str = str(glGetString(GL_SHADING_LANGUAGE_VERSION), 'utf-8')
        print('Loaded OpenGL {} with GLSL {}'.format(version_str, shader_version_str))

        # print('All supported GLSL versions:')
        # num_shading_versions = np.empty((1,), dtype=np.int32)
        # glGetIntegerv(GL_NUM_SHADING_LANGUAGE_VERSIONS, num_shading_versions)
        # print()
        # for i in range(num_shading_versions[0]):
        #     print(str(glGetStringi(GL_SHADING_LANGUAGE_VERSION, i), 'utf-8'))
        # print()

def make_vbo(data, usage, target, shader, attr_prefix, divisor=0):
    vbo = VBO(data, usage, target)
    with vbo:
        for prop, (dtype, offset) in data.dtype.fields.items():
            prop = attr_prefix + prop
            loc = glGetAttribLocation(shader, prop)
            size = int(np.prod(dtype.shape))
            stride = data.dtype.itemsize
            offset = c_void_p(offset)
            print('setting up vertex attribute "{}" @{:d}'.format(prop, loc))
            print('size:', size)
            print('stride:', stride)
            print('offset:', offset)
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(
                index=loc,
                size=size,
                type=GL_FLOAT,
                normalized=GL_FALSE,
                stride=stride,
                pointer=offset)
            glVertexAttribDivisor(loc, divisor)
        vbo.copy_data()
    return vbo
