"""utility module to provide api for setting the triangulation backend.

This module is intended to use if napari settings are not used.
If `set_backend` is used nex to settings, the state of the settings
may be inconsistent with the backend used.
"""

from enum import auto

from napari.utils.compat import StrEnum


class TriangulationBackend(StrEnum):
    """Enum-like class to specify which triangulation backend to use.

    All backends, except `pure_python` are using numba compiled helper functions
    """

    bermuda = auto()
    """Compiled backend implemented in Rust, https://github.com/napari/bermuda"""
    partsegcore = auto()
    """Compiled backend implemented in C++, https://partseg.github.io"""
    triangle = auto()
    """Triangle backend implemented in C, https://www.cs.cmu.edu/~quake/triangle.html"""
    numba = auto()
    """part of helper functions compiled with numba. Triangulation using vispy"""
    pure_python = auto()
    """Putre python on napari side, Triangulation using vispy"""

    def __str__(self) -> str:
        """Return the string representation of the backend."""
        return self.name.replace('_', ' ').title()

    def __repr__(self) -> str:
        """Return the string representation of the backend."""
        return self.name


def get_backend() -> TriangulationBackend:
    """Get the triangulation backend to use.

    Returns
    -------
    TriangulationBackend
        The triangulation backend to use.
    """
    from napari.layers.shapes._shapes_models import shape

    return shape.TRIANGULATION_BACKEND


def set_backend(backend: TriangulationBackend) -> TriangulationBackend:
    """Set the triangulation backend to use.

    Parameters
    ----------
    backend : TriangulationBackend
        The triangulation backend to use.

    Returns
    -------
    TriangulationBackend
        The previous triangulation backend.
    """
    from napari.layers.shapes._accelerated_triangulate_dispatch import (
        _set_numba,
    )
    from napari.layers.shapes._shapes_models import shape

    _set_numba(backend != TriangulationBackend.pure_python)

    prev = shape.TRIANGULATION_BACKEND

    shape.TRIANGULATION_BACKEND = backend
    return prev
