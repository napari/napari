# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import itertools
import os
from collections.abc import Callable
from contextlib import suppress
from functools import cache, wraps

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


class Shapes2DSuite:
    """Benchmarks for the Shapes layer with 2D data"""

    data: list[np.ndarray]
    layer: Shapes

    params = [2**i for i in range(4, 9)]

    if 'PR' in os.environ:
        skip_params = [(2**i,) for i in range(6, 9)]

    def setup(self, n):
        rng = np.random.default_rng(0)
        self.data = [50 * rng.random((6, 2)) for _ in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, _n):
        """Time to create an image layer."""
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
        for i in range(100):
            self.layer.get_value((i,) * 2)

    def mem_layer(self, _n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, _n):
        """Memory used by raw data."""
        return self.data


class Shapes3DSuite:
    """Benchmarks for the Shapes layer with 3D data."""

    data: list[np.ndarray]
    layer: Shapes

    params = [2**i for i in range(4, 9)]
    if 'PR' in os.environ:
        skip_params = [(2**i,) for i in range(6, 9)]

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


class _BackendSelection:
    triangulate: Callable | None
    prev_settings: bool

    def select_backend(self, compiled_triangulation):
        with suppress(AttributeError):
            self.prev_settings = (
                get_settings().experimental.compiled_triangulation
            )
            get_settings().experimental.compiled_triangulation = (
                compiled_triangulation
            )

        from napari.layers.shapes import _shapes_utils

        self.triangulate = _shapes_utils.triangulate
        _shapes_utils.triangulate = None
        self.warmup_numba()

    @staticmethod
    def warmup_numba() -> None:
        try:
            from napari.layers.shapes._accelerated_triangulate_dispatch import (
                warmup_numba_cache,
            )
        except ImportError:
            return
        warmup_numba_cache()

    def revert_backend(self):
        with suppress(AttributeError):
            get_settings().experimental.compiled_triangulation = (
                self.prev_settings
            )

        from napari.layers.shapes import _shapes_utils

        _shapes_utils.triangulate = self.triangulate

    def teardown(self, *_):
        self.revert_backend()


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
        'compiled_triangulation',
    ]
    params = [
        (
            100,
            5_000,
        ),
        (8, 32, 128),
        ('path', 'polygon'),
        (True, False),
    ]


class ShapeTriangulationNonConvexSuite(_ShapeTriangulationBaseShapeCount):
    skip_params = Skip(
        # skip a case when vispy triangulation backend fails
        always=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: n_shapes == 5000
        and n_points == 128
        and shape_type == 'polygon'
        and not compiled_triangulation
    )

    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = non_convex_no_self_intersection_polygons(
            n_shapes, n_points
        )
        self.select_backend(compiled_triangulation)


class ShapeTriangulationConvexSuite(_ShapeTriangulationBaseShapeCount):
    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = convex_polygons(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


class ShapeTriangulationIntersectionSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            5_000,
        ),
        (7, 9, 15, 33),
        ('path', 'polygon'),
        (True, False),
    ]
    skip_params = Skip(
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: not compiled_triangulation and n_shapes == 5000
    )

    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = self_intersecting_polygons(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


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
        (True, False),
    ]
    skip_params = Skip(
        always=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: n_shapes == 5000
        and n_points in {15, 33}
        and shape_type == 'polygon',
        if_on_ci=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: (
            # 7 and 9 points are too slow to run
            n_shapes == 5000
            and shape_type == 'polygon'
            and not compiled_triangulation
        )
        or (
            n_shapes == 100
            and n_points == 33
            and shape_type == 'polygon'
            and not compiled_triangulation
        ),
    )

    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = self_intersecting_stars_polygons(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


class ShapeTriangulationHoleSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            1_000,
        ),
        (12, 48),
        ('path', 'polygon'),
        (True, False),
    ]
    skip_params = Skip(
        if_in_pr=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: n_shapes > 100
    )

    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = polygons_with_hole(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


class ShapeTriangulationHolesSuite(_ShapeTriangulationBaseShapeCount):
    params = [
        (
            100,
            1_000,
        ),
        (24, 48),
        ('path', 'polygon'),
        (True, False),
    ]
    skip_params = Skip(
        if_in_pr=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: n_shapes > 100
    )

    def setup(self, n_shapes, n_points, _shape_type, compiled_triangulation):
        self.data = polygons_with_hole(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


class ShapeTriangulationMixed(_ShapeTriangulationBase):
    param_names = ['n_shapes', 'shape_type', 'compiled_triangulation']
    params = [
        (
            100,
            3_000,
        ),
        ('path', 'polygon'),
        (True, False),
    ]

    # the case of 128 points crashes the benchmark on call of PolygonData(vertices=data).triangulate()
    skip_params = Skip(
        if_in_pr=lambda n_shapes, shape_type, compiled_triangulation: n_shapes
        > 1000,
    )

    def setup(self, n_shapes, _shape_type, compiled_triangulation):
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
        self.select_backend(compiled_triangulation)

    @get_shape_type
    def time_create_shapes(self, shape_type):
        """Time to create a layer."""
        cls = shape_classes[shape_type]
        for data in self.data:
            cls(data)


class MeshTriangulationSuite(_BackendSelection):
    data: list[Polygon | Path]

    param_names = ['shape_type', 'compiled_triangulation']
    params = [('path', 'polygon'), (True, False)]

    def setup(self, shape_type, compiled_triangulation):
        self.select_backend(compiled_triangulation)
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

    def time_set_meshes(self, shape_type, _compiled_triangulation):
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
