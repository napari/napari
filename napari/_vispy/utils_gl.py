"""OpenGL Utilities.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Tuple

from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas


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


@lru_cache()
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

    # vispy doesn't expose GL_MAX_3D_TEXTURE_SIZE so estimating based on
    # memory requirements for 2D texture. This is not necessarily correct.
    # MAX_TEXTURE_SIZE_3D = gl.glGetParameter(gl.GL_MAX_3D_TEXTURE_SIZE)
    # if MAX_TEXTURE_SIZE_3D == ():
    #    MAX_TEXTURE_SIZE_3D = None
    if max_size_2d is not None:
        max_size_3d = round(max_size_2d ** (2.0 / 3.0))
    else:
        max_size_3d = None

    return max_size_2d, max_size_3d
