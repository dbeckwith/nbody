# -*- coding: utf-8 -*-

import ctypes

import numpy as np
from OpenGL.GL import *


def print_gl_version():
    # get OpenGL version strings, converting to UTF-8
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
    """
    Helper class representing an OpenGL buffer object.
    """

    def __init__(self, target):
        """
        Generates a new buffer.

        Arguments:
            target: enum, the OpenGL buffer target
        """
        self.target = target
        self._buf_id = glGenBuffers(1)

    def __enter__(self):
        """
        Context manager enter that binds the buffer.
        """
        glBindBuffer(self.target, self._buf_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit that unbinds the buffer.
        """
        glBindBuffer(self.target, 0)

        return False

class ConstBufferObject(BufferObject):
    """
    Helper class representing a buffer with constant data.
    """

    def __init__(self, usage, target, data):
        """
        Generates a new buffer.

        Arguments:
            usage: enum, the OpenGL buffer usage
            target: enum, the OpenGL buffer target
            data: np.ndarray, the data to initialize the buffer contents with
        """
        super().__init__(target)
        self.usage = usage
        self.data = data
        self.dtype = data.dtype
        self.length = len(data)

        # bind the buffer and buffer the given data
        with self:
            glBufferData(self.target, self.data.nbytes, self.data, self.usage)

class MappedBufferObject(BufferObject):
    """
    Helper class representing a buffer with memory-mapped storage.
    See http://www.bfilipek.com/2015/01/persistent-mapped-buffers-in-opengl.html
    """

    def __init__(self, target, dtype, length, flags):
        """
        Generates a new buffer.

        Arguments:
            target: enum, the OpenGL buffer target
            dtype: np.dtype, the element NumPy data type of the mapped array
            length: int in (0, inf), the number of elements to allocate
            flags: int, the flags to pass to glBufferStorage and glMapBufferRange
        """
        super().__init__(target)
        self.dtype = dtype
        self.length = length
        self.flags = flags

        # bind buffer
        with self:
            # number of bytes in array is dtype's itemsize times the length
            data_size = self.dtype.itemsize * self.length
            # initialize buffer storage with no data
            glBufferStorage(self.target, data_size, None, self.flags)
            # create a pointer to the mapped memory for the buffer
            ptr = glMapBufferRange(self.target, 0, data_size, self.flags)

            # next, create NumPy array from pointer, see http://stackoverflow.com/a/38606682
            # first, create a ctypes data type representing a float array
            # length of array is (buffer size in bytes) / (bytes per float)
            arr_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
            # cast pointer to array type and create raw NumPy array from it
            self.data = np.ctypeslib.as_array(arr_type.from_address(ptr))
            # create an np.ndarray view of the raw array with the given element type
            self.data = self.data.view(dtype=self.dtype, type=np.ndarray)

_gl_sync_obj = None

def gl_lock():
    """
    Creates an OpenGL lock that can be waited upon for GPU commands to complete.
    See http://www.bfilipek.com/2015/01/persistent-mapped-buffers-in-opengl.html, section "Syncing"
    """
    global _gl_sync_obj

    if _gl_sync_obj:
        glDeleteSync(_gl_sync_obj)

    _gl_sync_obj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

def gl_wait():
    """
    Waits on the previously create OpenGL lock.
    """
    global _gl_sync_obj

    if _gl_sync_obj:
        while True:
            wait_ret = glClientWaitSync(_gl_sync_obj, GL_SYNC_FLUSH_COMMANDS_BIT, 1)
            if wait_ret in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED):
                return

def gl_sync():
    """
    Waits for any pending OpenGL GPU commands to complete.
    """
    gl_lock()
    gl_wait()

def setup_vbo_attrs(vbo, shader, attr_prefix=None, divisor=0):
    """
    Creates all the necessary shader attribute bindings for the given VBO.

    Arguments:
        vbo: BufferObject, the VBO whose data attributes should be bound
        shader: OpenGL shader reference, the shader to get the attributes from
        attr_prefix: str, the name prefix for the attributes in the shader source
        divisor: int, the instancing divisor for these attributes, should be 1 if these attributes should be used per-instance
    """
    # bind the buffer as an array buffer no matter what kind it was inteded to be used as
    glBindBuffer(GL_ARRAY_BUFFER, vbo._buf_id)
    # go through the buffer's data type's fields
    for prop, (sub_dtype, offset) in vbo.dtype.fields.items():
        # prop is the name of the field
        # sub_dtype is the type of the field
        # offset is the offset in each element of the field

        # get the location of the attribute in the shader
        if attr_prefix is not None:
            prop = attr_prefix + prop
        loc = glGetAttribLocation(shader, prop)
        if loc == -1:
            print('WARNING: shader variable {:s} not found'.format(prop))
            continue

        # total size of the field is the product of the shape
        size = int(np.prod(sub_dtype.shape))

        # stride is number of bytes between each element
        stride = vbo.dtype.itemsize

        # convert offset to void pointer
        offset = ctypes.c_void_p(offset)

        # bind attributes
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(
            index=loc,
            size=size,
            type=GL_FLOAT,
            normalized=GL_FALSE,
            stride=stride,
            pointer=offset)
        glVertexAttribDivisor(loc, divisor)
    # unbind buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0)
