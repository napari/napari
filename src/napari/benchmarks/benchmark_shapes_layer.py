# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import itertools
from collections.abc import Callable
from contextlib import suppress
from enum import StrEnum, auto
from functools import cache, wraps
from typing import Any

import numpy as np

from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import shape_classes
from napari.layers.shapes._shapes_models import Path, Polygon
from napari.settings import get_settings
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

try:
    from .utils import Skip
except ImportError:
    from napari.benchmarks.utils import Skip

try:
    from napari.utils.triangulation_backend import TriangulationBackend
except ImportError:

    class TriangulationBackend(StrEnum):
        partsegcore = (
            auto()
        )  # data preprocessing using PartSegCore (https://partseg.github.io)
        bermuda = auto()  # data preprocessing using bermuda (https://github.com/napari/bermuda)
        triangle = auto()  # data preprocessing using numba, edge triangulation using numba, triangulation using triangle
        numba = auto()  # data preprocessing and edge triangulation using numba, triangulation using vispy
        pure_python = auto()  # data preprocessing, edge triangulation, and triangulation using python and vispy

        def __str__(self):
            return self.name

        def __repr__(self):
            return self.name


backends = list(TriangulationBackend)

backend_list_complex = [
    TriangulationBackend.partsegcore,
    TriangulationBackend.bermuda,
    TriangulationBackend.triangle,
    TriangulationBackend.numba,
]
# pure python backend is too slow for large datasets
# triangle backend requires deduplication of vertices in #6654


class _BackendSelection:
    """Provides a representation for a preferred backend if accelerated triangulation is unavailable."""

    triangulate: Callable | None  # a triangulate function
    prev_settings_old: bool  # save the state of NapariSettings.experimental.compiled_triangulation,
    prev_settings_new: Any  # save the state of NapariSettings.experimental.triangulation_backend,
    prev_numba: dict[
        str, Callable
    ]  # Maps numba function names to _accelerated_triangulate_dispatch functions.

    def _disable_numba(self):
        """Replace numba jit functions with pure python functions."""
        try:
            from napari.layers.shapes import (
                _accelerated_triangulate_dispatch as _triangle_dispatch,
            )
        except ImportError:
            return

        if not hasattr(_triangle_dispatch, '__all__'):
            # new dispatch do not require this
            return

        for name in _triangle_dispatch.__all__:
            # This part of the code assumes that all pure python functions
            # from _accelerated_triangulate_dispatch that are used when import from
            # _accelerated_triangulate module fails because of missing numba
            # are named as the function that they replace with added _py suffix.
            # So we could write generic code to replace all of them.
            if hasattr(_triangle_dispatch, f'{name}_py'):
                self.prev_numba[name] = getattr(_triangle_dispatch, name)
                setattr(
                    _triangle_dispatch,
                    name,
                    getattr(_triangle_dispatch, f'{name}_py'),
                )

    def _set_settings_old(self, triangulation_backend: TriangulationBackend):
        with suppress(AttributeError):
            self.prev_settings = get_settings().experimental.backend_type
            get_settings().experimental.backend_type = (
                triangulation_backend == TriangulationBackend.partsegcore
            )
            get_settings().experimental.compiled_triangulation = (
                triangulation_backend
                in {
                    TriangulationBackend.partsegcore,
                    TriangulationBackend.bermuda,
                }
            )

    def _set_settings_new(self, triangulation_backend: TriangulationBackend):
        self.prev_settings_new = (
            get_settings().experimental.triangulation_backend
        )
        get_settings().experimental.triangulation_backend = (
            triangulation_backend
        )

    def select_backend(self, triangulation_backend: TriangulationBackend):
        """Select a desired backend for triangulation."""
        self.prev_numba = {}
        if hasattr(get_settings().experimental, 'triangulation_backend'):
            self._set_settings_new(triangulation_backend)
        else:
            self._set_settings_old(triangulation_backend)

        from napari.layers.shapes import _shapes_utils

        self.triangulate = _shapes_utils.triangulate
        if triangulation_backend != TriangulationBackend.triangle:
            # Disable the triangle backend by overriding the function
            _shapes_utils.triangulate = None
        if triangulation_backend == TriangulationBackend.pure_python:
            self._disable_numba()
        else:
            self.warmup_numba()

    @staticmethod
    def warmup_numba() -> None:
        """Warmup numba cache to avoid the first call being slow."""
        try:
            from napari.layers.shapes._accelerated_triangulate_dispatch import (
                warmup_numba_cache,
            )
        except ImportError:
            return
        warmup_numba_cache()

    def revert_backend(self):
        """Restore changes made by select_backend function. Call in teardown step."""
        if hasattr(get_settings().experimental, 'triangulation_backend'):
            get_settings().experimental.triangulation_backend = (
                self.prev_settings_new
            )
        else:
            with suppress(AttributeError):
                get_settings().experimental.compiled_triangulation = (
                    self.prev_settings
                )

        from napari.layers.shapes import _shapes_utils

        _shapes_utils.triangulate = self.triangulate
        with suppress(ImportError):
            from napari.layers.shapes import (
                _accelerated_triangulate_dispatch as atd,
            )

            for name, func in self.prev_numba.items():
                setattr(atd, name, func)

    def teardown(self, *_):
        self.revert_backend()


def skip_above_100(n_shapes, *_):
    return n_shapes > 100


class Shapes2DSuite(_BackendSelection):
    """Benchmarks for the Shapes layer with 2D data"""

    data: list[np.ndarray]
    layer: Shapes

    params = [tuple(2**i for i in range(4, 9)), backends]
    params_names = ['n_shapes', 'backend']

    skip_params = Skip(if_in_pr=lambda n_shapes, backend: n_shapes > 2**5)

    def setup(self, n_shapes, backend):
        self.select_backend(backend)
        rng = np.random.default_rng(0)
        self.data = [50 * rng.random((6, 2)) for _ in range(n_shapes)]
        self.layer = Shapes(self.data, shape_type='polygon')
        self.layer.selected_data = list(range(n_shapes))

    def time_create_layer(self, *_):
        """Time to create an image layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, *_):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, *_):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, *_):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, *_):
        """Time to get current value."""
        for i in range(100):
            self.layer.get_value((i,) * 2)

    def mem_layer(self, *_):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, *_):
        """Memory used by raw data."""
        return self.data

    def time_edit_shape(self, *_):
        """Time to edit a shape."""
        # Simulate editing the first shape
        self.layer._data_view.edit(
            0, np.array([[10, 10], [20, 20], [30, 30]]), new_type='polygon'
        )


class Shape2DEditSuite:
    params = [tuple(2**i for i in range(4, 9)), ('single', 'all')]
    params_names = ['n_shapes', 'selection']

    def setup(self, n_shapes, selection):
        rng = np.random.default_rng(0)
        self.data = [50 * rng.random((6, 2)) for _ in range(n_shapes)]
        self.layer = Shapes(self.data, shape_type='polygon')
        if selection == 'all':
            self.layer.selected_data = list(range(n_shapes))
        else:
            self.layer.selected_data = [2]

    def time_set_edge_width(self, *_):
        self.layer.current_edge_width = 10

    def time_set_edge_color(self, *_):
        self.layer.current_edge_color = 'red'  # RGBA red

    def time_set_face_color(self, *_):
        self.layer.current_face_color = 'red'


class Shapes3DSuite:
    """Benchmarks for the Shapes layer with 3D data."""

    data: list[np.ndarray]
    layer: Shapes

    params = [2**i for i in range(4, 9)]
    skip_params = Skip(if_in_pr=lambda n_shapes: n_shapes > 2**5)

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.data = [50 * rng.random((6, 3)) for _ in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, _n):
        """Time to create a layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, _n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, _n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, _n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, _n):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def mem_layer(self, _n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, _n):
        """Memory used by raw data."""
        return self.data


class ShapesInteractionSuite:
    """Benchmarks for interacting with the Shapes layer with 2D data"""

    data: list[np.ndarray]
    layer: Shapes

    params = [2**i for i in range(4, 9)]

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.data = [50 * rng.random((6, 2)) for _ in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')
        self.layer.mode = 'select'

        # initialize the position and select a shape
        position = tuple(np.mean(self.layer.data[0], axis=0))

        # create events
        click_event = read_only_mouse_event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
        # Simulate click
        mouse_press_callbacks(self.layer, click_event)

        release_event = read_only_mouse_event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )

        # Simulate release
        mouse_release_callbacks(self.layer, release_event)

    def time_drag_shape(self, _n):
        """Time to process 5 shape drag events"""
        # initialize the position and select a shape
        position = tuple(np.mean(self.layer.data[0], axis=0))

        # create events
        click_event = read_only_mouse_event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )

        # Simulate click
        mouse_press_callbacks(self.layer, click_event)

        # create events
        drag_event = read_only_mouse_event(
            type='mouse_press',
            is_dragging=True,
            modifiers=[],
            position=position,
        )

        # start drag event
        mouse_move_callbacks(self.layer, drag_event)

        # simulate 5 drag events
        for _ in range(5):
            position = tuple(np.add(position, [10, 5]))
            drag_event = read_only_mouse_event(
                type='mouse_press',
                is_dragging=True,
                modifiers=[],
                position=position,
            )

            # Simulate move, click, and release
            mouse_move_callbacks(self.layer, drag_event)

        release_event = read_only_mouse_event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )

        # Simulate release
        mouse_release_callbacks(self.layer, release_event)

    time_drag_shape.param_names = ['n_shapes']

    def time_select_shape(self, _n):
        """Time to process shape selection events"""
        position = tuple(np.mean(self.layer.data[1], axis=0))

        # create events
        click_event = read_only_mouse_event(
            type='mouse_press',
            is_dragging=False,
            modifiers=[],
            position=position,
        )
        # Simulate click
        mouse_press_callbacks(self.layer, click_event)

        release_event = read_only_mouse_event(
            type='mouse_release',
            is_dragging=False,
            modifiers=[],
            position=position,
        )

        # Simulate release
        mouse_release_callbacks(self.layer, release_event)

    time_select_shape.param_names = ['n_shapes']


def get_shape_type(func):
    @wraps(func)
    def wrap(self, *args):
        shape_type_pos = self.param_names.index('shape_type')
        return func(self, args[shape_type_pos])

    return wrap


class _ShapeTriangulationBase(_BackendSelection):
    data: list[np.ndarray]

    @get_shape_type
    def time_create_layer(self, shape_type):
        """Time to create a Shapes layer when passed data and a shape type."""
        Shapes(self.data, shape_type=shape_type)


class _ShapeTriangulationBaseShapeCount(_ShapeTriangulationBase):
    param_names = [
        'n_shapes',
        'n_points',
        'shape_type',
        'triangulation_backend',
    ]
    params = [
        (
            100,
            5_000,
        ),
        (8, 32, 128),
        ('path', 'polygon'),
        backend_list_complex,
    ]

    skip_params = Skip(if_in_pr=skip_above_100)


class ShapeTriangulationNonConvexSuite(_ShapeTriangulationBaseShapeCount):
    skip_params = Skip(
        # skip a case when vispy triangulation backend fails
        always=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: n_points == 128
        and shape_type == 'polygon'
        and triangulation_backend != TriangulationBackend.triangle,
        if_in_pr=skip_above_100,
        # too slow (40 sec)
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: (
            n_shapes == 5000
            and n_points == 32
            and shape_type == 'polygon'
            and triangulation_backend
            in {
                TriangulationBackend.pure_python and TriangulationBackend.numba
            }
        ),
    )

    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = non_convex_no_self_intersection_polygons(
            n_shapes, n_points
        )
        self.select_backend(triangulation_backend)


class ShapeTriangulationConvexSuite(_ShapeTriangulationBaseShapeCount):
    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = convex_polygons(n_shapes, n_points)
        self.select_backend(triangulation_backend)


class ShapeTriangulationIntersectionSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            5_000,
        ),
        (7, 9, 15, 33),
        ('path', 'polygon'),
        backend_list_complex,
    ]
    skip_params = Skip(
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: triangulation_backend
        != TriangulationBackend.numba
        and n_shapes > 100,
        if_in_pr=skip_above_100,
    )

    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = self_intersecting_polygons(n_shapes, n_points)
        self.select_backend(triangulation_backend)


class ShapeTriangulationStarIntersectionSuite(
    _ShapeTriangulationBaseShapeCount
):
    params = [
        (
            100,
            5_000,
        ),
        (7, 9, 15, 33),
        ('path', 'polygon'),
        backend_list_complex,
    ]
    skip_params = Skip(
        always=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: n_shapes == 5000
        and n_points in {15, 33}
        and shape_type == 'polygon',
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: (
            (
                # 7 and 9 points are too slow to run
                n_shapes == 5000 and shape_type == 'polygon'
            )
            or (
                (n_shapes == 100 and n_points == 33)
                and shape_type == 'polygon'
                and triangulation_backend != TriangulationBackend.triangle
            )
        ),
        if_in_pr=skip_above_100,
    )

    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = self_intersecting_stars_polygons(n_shapes, n_points)
        self.select_backend(triangulation_backend)


def skip_triangle_backend(
    _n_shapes, _n_points, _shape_type, triangulation_backend
):
    """This is a helper function to skip triangle testing in polygon with holes case if
    the bugfix is not available.

    It may be removed after release on napari 0.6.0.
    """
    from napari.layers.shapes import _shapes_utils

    if hasattr(_shapes_utils, '_cull_triangles_not_in_poly'):
        return False
    return triangulation_backend == TriangulationBackend.triangle


class ShapeTriangulationHoleSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            1_000,
        ),
        (12, 48),
        ('path', 'polygon'),
        backend_list_complex,
    ]
    skip_params = Skip(
        always=skip_triangle_backend,
        if_in_pr=skip_above_100,
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: n_shapes > 100
        and triangulation_backend == TriangulationBackend.numba
        and shape_type == 'polygon',
    )

    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = polygons_with_hole(n_shapes, n_points)
        self.select_backend(triangulation_backend)


class ShapeTriangulationHolesSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            1_000,
        ),
        (24, 48),
        ('path', 'polygon'),
        backend_list_complex,
    ]
    skip_params = Skip(
        always=skip_triangle_backend,
        if_in_pr=skip_above_100,
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        triangulation_backend: n_shapes > 100
        and triangulation_backend == TriangulationBackend.numba
        and shape_type == 'polygon',
    )

    def setup(self, n_shapes, n_points, _shape_type, triangulation_backend):
        self.data = polygons_with_hole(n_shapes, n_points)
        self.select_backend(triangulation_backend)


class ShapeTriangulationMixed(_ShapeTriangulationBase):
    param_names = ['n_shapes', 'shape_type', 'triangulation_backend']
    params = [
        (
            100,
            3_000,
        ),
        ('path', 'polygon'),
        backend_list_complex,
    ]

    # the case of 128 points crashes the benchmark on call of PolygonData(vertices=data).triangulate()
    skip_params = Skip(if_in_pr=skip_above_100)

    def setup(self, n_shapes, _shape_type, triangulation_backend):
        part_size = int(n_shapes / 10)
        self.data = list(
            itertools.chain(
                convex_polygons(part_size, 4),
                convex_polygons(part_size * 2, 5),
                convex_polygons(part_size * 2, 7),
                convex_polygons(part_size, 60),
                non_convex_no_self_intersection_polygons(part_size, 10),
                non_convex_no_self_intersection_polygons(part_size, 60),
            )
        )
        self.select_backend(triangulation_backend)

    @get_shape_type
    def time_create_shapes(self, shape_type):
        """Time to create a layer."""
        cls = shape_classes[shape_type]
        for data in self.data:
            cls(data)


class MeshTriangulationSuite(_BackendSelection):
    data: list[Polygon | Path]

    param_names = ['shape_type', 'triangulation_backend']
    params = [('path', 'polygon'), backend_list_complex]

    def setup(self, shape_type, triangulation_backend):
        self.select_backend(triangulation_backend)
        part_size = 10
        self.data = [
            shape_classes[shape_type](x)
            for x in itertools.chain(
                convex_polygons(part_size, 4),
                convex_polygons(part_size * 2, 5),
                convex_polygons(part_size * 2, 12),
                convex_polygons(part_size, 60),
                non_convex_no_self_intersection_polygons(part_size, 10),
                non_convex_no_self_intersection_polygons(part_size, 60),
                self_intersecting_stars_polygons(part_size, 11),
                self_intersecting_polygons(part_size, 15),
            )
        ]

    def time_set_meshes(self, shape_type, _triangulation_backend):
        for shape in self.data:
            shape._set_meshes(
                shape.data,
                face=(shape_type == 'polygon'),
                closed=True,
                edge=True,
            )


@cache
def non_convex_no_self_intersection_polygons(
    n_shapes=5_000, n_points=32
) -> list[np.ndarray]:
    """
    Create a set of non-convex coordinates

    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_points : int
        Number of virtex of each shape
    """
    rng = np.random.default_rng(0)
    radius = 1000
    center = rng.uniform(1500, 4500, (n_shapes, 2))
    phi = np.linspace(0, 2 * np.pi, n_points)
    rays = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    rays = rays.reshape((1, -1, 2))
    rays = rays * rng.uniform(0.9, 1.1, (n_shapes, n_points, 2))
    center = center.reshape((-1, 1, 2))
    return list(center + radius * rays)


@cache
def self_intersecting_stars_polygons(
    n_shapes=5_000, n_points=31
) -> list[np.ndarray]:
    """
    Create a set of non-convex coordinates

    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_points : int
        Number of virtex of each shape
    """
    assert n_points % 2 == 1
    rng = np.random.default_rng(0)
    radius = 5000
    center = rng.uniform(5000, 15000, (n_shapes, 2))
    shift = np.floor(n_points / 2) + 1
    phi = np.linspace(0, 2 * np.pi, n_points + 1) * shift
    rays = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    rays = rays.reshape((1, -1, 2))
    rays = rays * rng.uniform(0.9, 1.1, (n_shapes, n_points + 1, 2))
    center = center.reshape((-1, 1, 2))
    return list(center + radius * rays)


@cache
def self_intersecting_polygons(
    n_shapes=5_000, n_points=31
) -> list[np.ndarray]:
    """
    Create a set of non-convex coordinates

    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_points : int
        Number of virtex of each shape
    """
    assert n_points % 2 == 1
    rng = np.random.default_rng(0)
    radius = 5000
    center = rng.uniform(5000, 15000, (n_shapes, 2))
    phi = np.linspace(0, 2 * np.pi, n_points + 1) * 2
    rays = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    rays = rays.reshape((1, -1, 2))
    rays = rays * rng.uniform(0.9, 1.1, (n_shapes, n_points + 1, 2))
    center = center.reshape((-1, 1, 2))
    return list(center + radius * rays)


@cache
def convex_polygons(n_shapes=5_000, n_points=32) -> list[np.ndarray]:
    """
    Create a set of convex coordinates

    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_points : int
        Number of virtex of each shape
    """
    rng = np.random.default_rng(0)
    radius = 500 / np.sqrt(n_shapes)
    center = rng.uniform(500, 1500, (n_shapes, 2))
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    rays = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    rays = rays.reshape((1, -1, 2))
    center = center.reshape((-1, 1, 2))
    return list(center + radius * rays)


@cache
def polygons_with_hole(n_shapes=5_000, n_points=32) -> list[np.ndarray]:
    """
    Create a set of polygon with hole

    Parameters
    ----------
    n_shapes : int
        Number of shapes to create
    n_points : int
        Number of virtex of each shape
    """
    rng = np.random.default_rng(0)
    assert n_points > 7, (
        'n_points should be greater than 7 to generate a polygon with holes'
    )
    radius = 500
    outer_points_num = n_points // 2 + 1
    inner_points_num = n_points - outer_points_num + 2

    center = rng.uniform(500, 1500, (n_shapes, 2))
    phi1 = np.linspace(0, 2 * np.pi, outer_points_num, endpoint=True)
    phi1[-1] = 0
    phi2 = np.linspace(0, 2 * np.pi, inner_points_num, endpoint=True)
    phi2[-1] = 0
    rays1 = np.stack([np.sin(phi1), np.cos(phi1)], axis=1) * radius
    rays2 = np.stack([np.sin(phi2), np.cos(phi2)], axis=1) * radius // 2
    rays = np.concatenate([rays1, rays2]).reshape((1, -1, 2))
    center = center.reshape((-1, 1, 2))
    return list(center + rays)


@cache
def polygons_with_holes(n_shapes=5_000, n_points=32) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    assert n_points > 20, (
        'n_points should be greater than 7 to generate a polygon with holes'
    )
    radius = 500
    inner_points_num = 4
    outer_points_num = n_points - 4 * inner_points_num

    rectangle = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]) * 50

    center = rng.uniform(500, 1500, (n_shapes, 2))
    phi = np.linspace(0, 2 * np.pi, outer_points_num, endpoint=True)
    phi[-1] = 0
    rays = np.stack([np.sin(phi), np.cos(phi)], axis=1) * radius
    steep = len(phi) // 4

    points = np.concatenate(
        [
            rays[:steep],
            rectangle + (250, 0),
            rays[steep - 1 : 2 * steep],
            rectangle + (0, -250),
            rays[2 * steep - 1 : 3 * steep],
            rectangle + (-250, 0),
            rays[3 * steep - 1 :],
            rectangle + (0, 250),
        ]
    ).reshape((1, -1, 2))

    center = center.reshape((-1, 1, 2))
    return list(center + points)


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
