# -*- coding: utf-8 -*-

import ctypes

import numpy as np
from OpenGL.GL import *


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

class BufferObject(object):
    def __init__(self, target):
        self.target = target
        self._buf_id = glGenBuffers(1)

    def __enter__(self):
        glBindBuffer(self.target, self._buf_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        glBindBuffer(self.target, 0)

        return False

class ConstBufferObject(BufferObject):
    def __init__(self, usage, target, data):
        super().__init__(target)
        self.usage = usage
        self.data = data
        self.dtype = data.dtype
        self.length = len(data)

        with self:
            glBufferData(self.target, self.data.nbytes, self.data, self.usage)

class MappedBufferObject(BufferObject):
    # http://www.bfilipek.com/2015/01/persistent-mapped-buffers-in-opengl.html

    def __init__(self, target, dtype, length, flags):
        super().__init__(target)
        self.dtype = dtype
        self.length = length
        self.flags = flags

        with self:
            data_size = self.dtype.itemsize * self.length
            glBufferStorage(self.target, data_size, None, self.flags)
            ptr = glMapBufferRange(self.target, 0, data_size, self.flags)
            arr_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
            self.data = np.ctypeslib.as_array(arr_type.from_address(ptr))
            self.data = self.data.view(dtype=self.dtype, type=np.ndarray)

_gl_sync_obj = None

def gl_lock():
    global _gl_sync_obj

    if _gl_sync_obj:
        glDeleteSync(_gl_sync_obj)

    _gl_sync_obj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

def gl_wait():
    global _gl_sync_obj

    if _gl_sync_obj:
        while True:
            wait_ret = glClientWaitSync(_gl_sync_obj, GL_SYNC_FLUSH_COMMANDS_BIT, 1)
            if wait_ret in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                return

def gl_sync():
    gl_lock()
    gl_wait()

def setup_vbo_attrs(vbo, shader, attr_prefix, divisor=0):
    glBindBuffer(GL_ARRAY_BUFFER, vbo._buf_id)
    for prop, (sub_dtype, offset) in vbo.dtype.fields.items():
        prop = attr_prefix + prop
        loc = glGetAttribLocation(shader, prop)
        if loc == -1:
            print('WARNING: shader variable {:s} not found'.format(prop))
            continue
        size = int(np.prod(sub_dtype.shape))
        stride = vbo.dtype.itemsize
        offset = ctypes.c_void_p(offset)
        # print('setting up vertex attribute "{}" @{:d}'.format(prop, loc))
        # print('size:', size)
        # print('stride:', stride)
        # print('offset:', offset)
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(
            index=loc,
            size=size,
            type=GL_FLOAT,
            normalized=GL_FALSE,
            stride=stride,
            pointer=offset)
        glVertexAttribDivisor(loc, divisor)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
