"""Module providing a unified interface for triangulation helper functions.

We don't want numba to be a required dependency. Therefore, for
numba-accelerated functions, we provide slower NumPy-only alternatives.

With this module, downstream modules can import these helper functions without
knowing which implementation is being used.
"""

from typing import Any

import numpy as np

from napari.layers.shapes import _accelerated_triangulate_python

try:
    from napari.layers.shapes import _accelerated_triangulate_numba
except ImportError:
    _accelerated_triangulate_numba = None

USE_NUMBA_FOR_EDGE_TRIANGULATION = _accelerated_triangulate_numba is not None
CACHE_WARMUP = False

normalize_vertices_and_edges = (
    _accelerated_triangulate_python.normalize_vertices_and_edges_py
)


def __getattr__(name: str) -> Any:
    if name in {'remove_path_duplicates', 'create_box_from_bounding'}:
        if _accelerated_triangulate_numba is not None:
            # If numba is available, use the numba implementation
            return getattr(_accelerated_triangulate_numba, name)
        # Otherwise, use the pure python implementation
        return getattr(_accelerated_triangulate_python, f'{name}_py')

    if USE_NUMBA_FOR_EDGE_TRIANGULATION:
        # If numba is available, use the numba implementation
        return getattr(_accelerated_triangulate_numba, name)
    # Otherwise, use the pure python implementation
    return getattr(_accelerated_triangulate_python, f'{name}_py')


def _set_numba(value: bool) -> None:
    """Set the Numba backend to use.

    Parameters
    ----------
    value : bool
        If True, use the Numba backend. If False, use the pure Python backend.
    """
    global USE_NUMBA_FOR_EDGE_TRIANGULATION
    USE_NUMBA_FOR_EDGE_TRIANGULATION = value and (
        _accelerated_triangulate_numba is not None
    )


def warmup_numba_cache() -> None:
    if _accelerated_triangulate_numba is None:
        # no numba, nothing to warm up
        return
    global CACHE_WARMUP
    if CACHE_WARMUP:
        return

    CACHE_WARMUP = True
    for order in ('C', 'F'):
        data = np.array(
            [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32, order=order
        )
        data2 = np.array([[1, 1], [10, 15]], dtype=np.float32, order=order)

        if not USE_NUMBA_FOR_EDGE_TRIANGULATION:
            _accelerated_triangulate_numba.generate_2D_edge_meshes(data, True)
            _accelerated_triangulate_numba.generate_2D_edge_meshes(data, False)
            _accelerated_triangulate_numba.remove_path_duplicates(data, False)
            _accelerated_triangulate_numba.is_convex(data)
            v, e = _accelerated_triangulate_numba.normalize_vertices_and_edges(
                data
            )
            _accelerated_triangulate_numba.reconstruct_polygons_from_edges(
                v, e
            )
        _accelerated_triangulate_numba.remove_path_duplicates(data, False)
        _accelerated_triangulate_numba.remove_path_duplicates(data, True)
        _accelerated_triangulate_numba.create_box_from_bounding(data2)
