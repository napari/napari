"""OpenGL Utilities.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Tuple

import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas

from ...utils.translations import trans

texture_dtypes = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


@contextmanager
def _opengl_context():
    """Assure we are running with a valid OpenGL context.

    Only create a Canvas is one doesn't exist. Creating and closing a
    Canvas causes vispy to process Qt events which can cause problems.
    Ideally call opengl_context() on start after creating your first
    Canvas. However it will work either way.
    """
    canvas = Canvas(show=False) if get_current_canvas() is None else None
    try:
        yield
    finally:
        if canvas is not None:
            canvas.close()


@lru_cache(maxsize=1)
def get_gl_extensions() -> str:
    """Get basic info about the Gl capabilities of this machine"""
    with _opengl_context():
        return gl.glGetParameter(gl.GL_EXTENSIONS)


@lru_cache
def get_max_texture_sizes() -> Tuple[int, int]:
    """Return the maximum texture sizes for 2D and 3D rendering.

    If this function is called without an OpenGL context it will create a
    temporary non-visible Canvas. Either way the lru_cache means subsequent
    calls to thing function will return the original values without
    actually running again.

    Returns
    -------
    Tuple[int, int]
        The max textures sizes for (2d, 3d) rendering.
    """
    with _opengl_context():
        max_size_2d = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)

    if max_size_2d == ():
        max_size_2d = None

    # vispy/gloo doesn't provide the GL_MAX_3D_TEXTURE_SIZE location,
    # but it can be found in this list of constants
    # http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.GL.html
    with _opengl_context():
        GL_MAX_3D_TEXTURE_SIZE = 32883
        max_size_3d = gl.glGetParameter(GL_MAX_3D_TEXTURE_SIZE)

    if max_size_3d == ():
        max_size_3d = None

    return max_size_2d, max_size_3d


def fix_data_dtype(data):
    """Makes sure the dtype of the data is accetpable to vispy.

    Acceptable types are int8, uint8, int16, uint16, float32.

    Parameters
    ----------
    data : np.ndarray
        Data that will need to be of right type.

    Returns
    -------
    np.ndarray
        Data that is of right type and will be passed to vispy.
    """

    dtype = np.dtype(data.dtype)
    if dtype in texture_dtypes:
        return data
    else:
        try:
            dtype = dict(i=np.float32, f=np.float32, u=np.uint16, b=np.uint8)[
                dtype.kind
            ]
        except KeyError:  # not an int or float
            raise TypeError(
                trans._(
                    'type {dtype} not allowed for texture; must be one of {textures}',  # noqa: E501
                    deferred=True,
                    dtype=dtype,
                    textures=set(texture_dtypes),
                )
            )
        return data.astype(dtype)


BLENDING_MODES = {
    'opaque': dict(
        depth_test=True,
        cull_face=False,
        blend=False,
        blend_func=('one', 'zero'),
        blend_equation='func_add',
    ),
    'translucent': dict(
        depth_test=True,
        cull_face=False,
        blend=True,
        blend_func=('src_alpha', 'one_minus_src_alpha', 'zero', 'one'),
        blend_equation='func_add',
    ),
    'translucent_no_depth': dict(
        depth_test=False,
        cull_face=False,
        blend=True,
        blend_func=('src_alpha', 'one_minus_src_alpha', 'zero', 'one'),
        blend_equation='func_add',  # see vispy/vispy#2324
    ),
    'additive': dict(
        depth_test=False,
        cull_face=False,
        blend=True,
        blend_func=('src_alpha', 'one'),
        blend_equation='func_add',
    ),
    'minimum': dict(
        depth_test=False,
        cull_face=False,
        blend=True,
        blend_func=('one', 'one'),
        blend_equation='min',
    ),
}
