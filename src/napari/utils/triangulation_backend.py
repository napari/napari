"""utility module to provide api for setting the triangulation backend.

This module is intended to use if napari settings are not used.
If `set_backend` is used nex to settings, the state of the settings
may be inconsistent with the backend used.
"""

import sys

from napari.utils.compat import StrEnum


class TriangulationBackend(StrEnum):
    """Enum-like class to specify which triangulation backend to use.

    All backends, except `pure_python` are using numba compiled helper functions
    """

    fastest_available = 'Fastest available'
    """Select the fastest available backend. The order of preference is:
    bermuda, partsegcore, triangle, numba, pure_python.
    """
    bermuda = 'bermuda'
    """Compiled backend implemented in Rust, https://github.com/napari/bermuda"""
    partsegcore = 'PartSegCore'
    """Compiled backend implemented in C++, https://partseg.github.io"""
    triangle = 'triangle'
    """Triangle backend implemented in C, https://www.cs.cmu.edu/~quake/triangle.html"""
    numba = 'Numba'
    """part of helper functions compiled with numba. Triangulation using vispy"""
    pure_python = 'Pure python'
    """Putre python on napari side, Triangulation using vispy"""

    def __str__(self) -> str:
        """Return the string representation of the backend."""
        return str(self.value)

    def __repr__(self) -> str:
        """Return the string representation of the backend."""
        return self.name

    @classmethod
    def _missing_(cls, value: object) -> 'TriangulationBackend':
        """Handle missing values in the enum."""
        # Handle the case where the value is not a valid enum member
        if isinstance(value, str):
            return cls[value.replace(' ', '_').lower()]
        raise ValueError(f"'{value}' is not a valid TriangulationBackend.")


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
        _set_warmup,
    )
    from napari.layers.shapes._shapes_models import shape

    bermuda_loaded = 'bermuda' in sys.modules
    partsegcore_loaded = 'PartSegCore_compiled_backend' in sys.modules

    any_compiled_loaded = bermuda_loaded or partsegcore_loaded
    # triangulation do not contain utils for edge triangulation

    need_numba_warmup = (
        backend == TriangulationBackend.numba
        or (
            backend == TriangulationBackend.fastest_available
            and not any_compiled_loaded
        )
        or (backend == TriangulationBackend.bermuda and not bermuda_loaded)
        or (
            backend == TriangulationBackend.partsegcore
            and not partsegcore_loaded
        )
    )

    _set_numba(backend != TriangulationBackend.pure_python)
    _set_warmup(need_numba_warmup)

    prev = shape.TRIANGULATION_BACKEND

    shape.TRIANGULATION_BACKEND = backend
    return prev
