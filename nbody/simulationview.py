# -*- coding: utf-8 -*-

import numpy as np
from pyrr import Vector3, Vector4, Matrix44

import OpenGL
print(dir(OpenGL))
OpenGL.ERROR_CHECKING = True
OpenGL.FULL_LOGGING = True
from OpenGL.GL import *
from OpenGL.GL import shaders

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QOpenGLWidget


def print_gl_version():
        version_str = str(glGetString(GL_VERSION), 'utf-8')
        shader_version_str = str(glGetString(GL_SHADING_LANGUAGE_VERSION), 'utf-8')
        print('Loaded OpenGL {} with GLSL {}'.format(version_str, shader_version_str))

        print('All supported GLSL versions:')
        num_shading_versions = np.empty((1,), dtype=np.int32)
        glGetIntegerv(GL_NUM_SHADING_LANGUAGE_VERSIONS, num_shading_versions)
        print()
        for i in range(num_shading_versions[0]):
            print(str(glGetStringi(GL_SHADING_LANGUAGE_VERSION, i), 'utf-8'))
        print()

class SimulationView(QOpenGLWidget):
    def __init__(self, parent):
        super().__init__(parent)

    def sizeHint(self):
        return QSize(800, 500)

    def initializeGL(self):
        print_gl_version()

    def paintGL(self):
        pass

    def resizeGL(self, width, height):
        pass
