"""Module providing a unified interface for triangulation helper functions.

We don't want numba to be a required dependency. Therefore, for
numba-accelerated functions, we provide slower NumPy-only alternatives.

With this module, downstream modules can import these helper functions without
knowing which implementation is being used.
"""

from types import ModuleType

import numpy as np

from napari.layers.shapes import _accelerated_triangulate_python

_accelerated_triangulate_numba: ModuleType | None

try:
    from napari.layers.shapes import _accelerated_triangulate_numba
except ImportError:
    _accelerated_triangulate_numba = None

USE_NUMBA_FOR_EDGE_TRIANGULATION = _accelerated_triangulate_numba is not None
RUN_WARMUP = _accelerated_triangulate_numba is not None
CACHE_WARMUP = False
# Some numba functions are used even if for non-numba triangulation backends,
# *if* numba is available. This tracks whether those functions have been warmed
# up already.
UNIVERSAL_CACHE_WARMUP = False

normalize_vertices_and_edges = (
    _accelerated_triangulate_python.normalize_vertices_and_edges_py
)


ALWAYS_NUMBA = ('remove_path_duplicates', 'create_box_from_bounding')
SWAPPABLE_NUMBA = (
    'generate_2D_edge_meshes',
    'is_convex',
    'normalize_vertices_and_edges',
    'reconstruct_polygons_from_edges',
)

if _accelerated_triangulate_numba is not None:
    remove_path_duplicates = (
        _accelerated_triangulate_numba.remove_path_duplicates
    )
    create_box_from_bounding = (
        _accelerated_triangulate_numba.create_box_from_bounding
    )
    generate_2D_edge_meshes = (
        _accelerated_triangulate_numba.generate_2D_edge_meshes
    )
    is_convex = _accelerated_triangulate_numba.is_convex
    normalize_vertices_and_edges = (
        _accelerated_triangulate_numba.normalize_vertices_and_edges
    )
    reconstruct_polygons_from_edges = (
        _accelerated_triangulate_numba.reconstruct_polygons_from_edges
    )

else:
    remove_path_duplicates = (
        _accelerated_triangulate_python.remove_path_duplicates_py
    )
    create_box_from_bounding = (
        _accelerated_triangulate_python.create_box_from_bounding_py
    )
    generate_2D_edge_meshes = (
        _accelerated_triangulate_python.generate_2D_edge_meshes_py
    )
    is_convex = _accelerated_triangulate_python.is_convex_py
    reconstruct_polygons_from_edges = (
        _accelerated_triangulate_python.reconstruct_polygons_from_edges_py
    )


def _set_numba(value: bool) -> None:
    """Set the Numba backend to use.

    Parameters
    ----------
    value : bool
        If True, use the Numba backend. If False, use the pure Python backend.
    """
    global USE_NUMBA_FOR_EDGE_TRIANGULATION

    val = value and (_accelerated_triangulate_numba is not None)
    if val:
        for name in SWAPPABLE_NUMBA:
            globals()[name] = getattr(_accelerated_triangulate_numba, name)
    else:
        for name in SWAPPABLE_NUMBA:
            globals()[name] = getattr(
                _accelerated_triangulate_python, f'{name}_py'
            )

    USE_NUMBA_FOR_EDGE_TRIANGULATION = val


def _set_warmup(value: bool) -> None:
    global RUN_WARMUP
    RUN_WARMUP = value


def warmup_universal_numba() -> None:
    """Warm up the functions that are used even with compiled backends."""
    if _accelerated_triangulate_numba is None:
        # no numba, nothing to warm up
        return
    global UNIVERSAL_CACHE_WARMUP
    if UNIVERSAL_CACHE_WARMUP:
        return
    UNIVERSAL_CACHE_WARMUP = True

    for order in ('C', 'F'):
        data = np.array(
            [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32, order=order
        )
        data2 = np.array([[1, 1], [10, 15]], dtype=np.float32, order=order)

        _accelerated_triangulate_numba.remove_path_duplicates(data, False)
        _accelerated_triangulate_numba.remove_path_duplicates(data, True)
        _accelerated_triangulate_numba.create_box_from_bounding(data2)


def warmup_numba_cache() -> None:
    if _accelerated_triangulate_numba is None:
        # no numba, nothing to warm up
        return
    if not RUN_WARMUP:
        warmup_universal_numba()
        return
    global CACHE_WARMUP
    if CACHE_WARMUP:
        return

    CACHE_WARMUP = True
    warmup_universal_numba()

    for order in ('C', 'F'):
        data = np.array(
            [[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32, order=order
        )

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
