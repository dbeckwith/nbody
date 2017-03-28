# -*- coding: utf-8 -*-

import os

import numpy as np
from pyrr import Vector3, Vector4, Matrix44

import OpenGL
OpenGL.ERROR_CHECKING = True
OpenGL.FULL_LOGGING = True
from OpenGL.GL import *
from OpenGL.GL import shaders

from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtWidgets import QOpenGLWidget


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

class SimulationView(QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.size = QSize(800, 500)

        self.camera = Camera(
            eye = Vector3([0.0, 0.0, -10.0]),
            at = Vector3([0.0, 0.0, 0.0]),
            up = Vector3([0.0, -1.0, 0.0]),
            fovx = np.deg2rad(90.0),
            aspect = self.size.width() / self.size.height(),
            near = 0.01,
            far = 100.0)

    def sizeHint(self):
        return self.size

    def initializeGL(self):
        print_gl_version()

        glShadeModel(GL_FLAT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        with open(os.path.join(os.path.dirname(__file__), 'sim.vert'), 'r') as f:
            vshader = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
        with open(os.path.join(os.path.dirname(__file__), 'sim.frag'), 'r') as f:
            fshader = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vshader, fshader)

        self.mvp_loc = glGetUniformLocation(self.shader, 'mvp')

        self.verticies = np.array(
            [[-1., -1., 0.],
             [ 1., -1., 0.],
             [ 1.,  1., 0.]])
        self.verticies = self.verticies.astype(np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.verticies.nbytes, None, GL_STREAM_DRAW)

        position_loc = glGetAttribLocation(self.shader, 'position')
        glEnableVertexAttribArray(position_loc)
        glVertexAttribPointer(
            index=position_loc,
            size=self.verticies.shape[-1],
            type=GL_FLOAT,
            normalized=GL_FALSE,
            stride=0,
            pointer=None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.ani_timer = QTimer(self)
        self.ani_timer.setInterval(1000 / 60)
        self.ani_timer.timeout.connect(self.update)
        self.ani_timer.start()

    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        mvp = self.camera.view_proj.astype(np.float32)
        glUniformMatrix4fv(self.mvp_loc, 1, GL_FALSE, mvp)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.verticies.nbytes, None, GL_STREAM_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.verticies.nbytes, self.verticies)

        glDrawArrays(GL_TRIANGLES, 0, self.verticies.shape[0])

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

        self.verticies[0, 0] -= 0.01

    def resizeGL(self, width, height):
        self.size = QSize(width, height)
        self.camera.aspect = self.size.width() / self.size.height()
        self.camera.update()

class Camera(object):
    def __init__(self, eye, at, up, fovx, aspect, near, far):
        self.eye = eye
        self.at = at
        self.up = up
        self.fovx = fovx
        self.aspect = aspect
        self.near = near
        self.far = far

        self._view_proj = None

    @property
    def view_proj(self):
        if self._view_proj is None:
            eye = self.eye
            at = self.at
            up = self.up

            n = (at - eye).normalised
            u = up.normalised.cross(n)
            v = n.cross(u)
            view = Matrix44(
                [[u.x, u.y, u.z, -u.dot(eye)],
                 [v.x, v.y, v.z, -v.dot(eye)],
                 [n.x, n.y, n.z, -n.dot(eye)],
                 [  0,   0,   0,           1]])

            fovx = self.fovx
            aspect = self.aspect
            near = self.near
            far = self.far

            fov_tan_inv = 1 / np.tan(fovx / 2)
            a = fov_tan_inv
            b = aspect * fov_tan_inv
            c = -(far + near) / (far - near)
            d = -2 * far * near / (far - near)
            proj = Matrix44(
                [[a, 0,  0, 0],
                 [0, b,  0, 0],
                 [0, 0,  c, d],
                 [0, 0, -1, 0]])

            self._view_proj = proj * view

        return self._view_proj

    def update(self):
        self._view_proj = None
