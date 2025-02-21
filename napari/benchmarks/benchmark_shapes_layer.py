# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import itertools
import os
from contextlib import suppress
from functools import cache, wraps

import numpy as np

from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import shape_classes
from napari.settings import get_settings
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from .utils import Skip


class Shapes2DSuite:
    """Benchmarks for the Shapes layer with 2D data"""

    params = [2**i for i in range(4, 9)]

    if 'PR' in os.environ:
        skip_params = [(2**i,) for i in range(6, 9)]

    def setup(self, n):
        np.random.seed(0)
        self.data = [50 * np.random.random((6, 2)) for i in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, n):
        """Time to create an image layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        for i in range(100):
            self.layer.get_value((i,) * 2)

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Shapes3DSuite:
    """Benchmarks for the Shapes layer with 3D data."""

    params = [2**i for i in range(4, 9)]
    if 'PR' in os.environ:
        skip_params = [(2**i,) for i in range(6, 9)]

    def setup(self, n):
        np.random.seed(0)
        self.data = [50 * np.random.random((6, 3)) for i in range(n)]
        self.layer = Shapes(self.data, shape_type='polygon')

    def time_create_layer(self, n):
        """Time to create a layer."""
        Shapes(self.data, shape_type='polygon')

    def time_refresh(self, n):
        """Time to refresh view."""
        self.layer.refresh()

    def time_set_view_slice(self, n):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_update_thumbnail(self, n):
        """Time to update thumbnail."""
        self.layer._update_thumbnail()

    def time_get_value(self, n):
        """Time to get current value."""
        self.layer.get_value((0,) * 3)

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class ShapesInteractionSuite:
    """Benchmarks for interacting with the Shapes layer with 2D data"""

    params = [2**i for i in range(4, 9)]

    def setup(self, n):
        np.random.seed(0)
        self.data = [50 * np.random.random((6, 2)) for i in range(n)]
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

    def time_drag_shape(self, n):
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

    def time_select_shape(self, n):
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


class _ShapeTriangulationBase:
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

    def revert_backend(self):
        with suppress(AttributeError):
            get_settings().experimental.compiled_triangulation = (
                self.prev_settings
            )

        from napari.layers.shapes import _shapes_utils

        _shapes_utils.triangulate = self.triangulate

    def teardown(self, *_):
        self.revert_backend()

    @get_shape_type
    def time_create_layer(self, shape_type):
        """Time to create a layer."""
        Shapes(self.data, shape_type=shape_type)

    @get_shape_type
    def time_create_shapes(self, shape_type):
        """Time to create a layer."""
        cls = shape_classes[shape_type]
        for data in self.data:
            cls(data)


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
        always=lambda n_shapes,
        n_points,
        shape_type,
        compiled_triangulation: n_shapes == 5000
        and n_points == 128
        and shape_type == 'polygon'
        and not compiled_triangulation
    )

    def setup(self, n_shapes, n_points, shape_type, compiled_triangulation):
        self.data = non_convex_cords(n_shapes, n_points)[4151:]
        self.select_backend(compiled_triangulation)


class ShapeTriangulationConvexSuite(_ShapeTriangulationBaseShapeCount):
    def setup(self, n_shapes, n_points, shape_type, compiled_triangulation):
        self.data = convex_cords(n_shapes, n_points)
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

    def setup(self, n_shapes, n_points, shape_type, compiled_triangulation):
        self.data = self_intersecting_cords(n_shapes, n_points)
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

    def setup(self, n_shapes, n_points, shape_type, compiled_triangulation):
        self.data = self_intersecting_stars_cords(n_shapes, n_points)
        self.select_backend(compiled_triangulation)


class ShapeTriangulationMixed(_ShapeTriangulationBase):
    param_names = ['n_shapes', 'shape_type', 'compiled_triangulation']
    params = [
        (
            100,
            5_000,
        ),
        ('path', 'polygon'),
        (True, False),
    ]

    # the case of 128 points crashes the benchmark on call of PolygonData(vertices=data).triangulate()
    skip_params = Skip(
        if_in_pr=lambda n_shapes, shape_type, compiled_triangulation: n_shapes
        > 1000,
    )

    def setup(self, n_shapes, shape_type, compiled_triangulation):
        part_size = int(n_shapes / 10)
        self.data = list(
            itertools.chain(
                convex_cords(part_size, 4),
                convex_cords(part_size * 2, 5),
                convex_cords(part_size * 2, 7),
                convex_cords(part_size, 60),
                non_convex_cords(part_size, 10),
                non_convex_cords(part_size, 60),
            )
        )
        self.select_backend(compiled_triangulation)


@cache
def non_convex_cords(n_shapes=5_000, n_points=32):
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
    return center + radius * rays


@cache
def self_intersecting_stars_cords(n_shapes=5_000, n_points=31):
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
    return center + radius * rays


@cache
def self_intersecting_cords(n_shapes=5_000, n_points=31):
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
    return center + radius * rays


@cache
def convex_cords(n_shapes=5_000, n_points=32):
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
    return center + radius * rays


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
