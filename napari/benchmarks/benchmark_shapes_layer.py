# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
# or the napari documentation on benchmarking
# https://github.com/napari/napari/blob/main/docs/BENCHMARKS.md
import os

import numpy as np

from napari.layers import Shapes
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)


class Shapes2DSuite:
    """Benchmarks for the Shapes layer with 2D data"""

    params = [2**i for i in range(4, 9)]

    if "PR" in os.environ:
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
        self.layer.get_value((0,) * 2)

    def mem_layer(self, n):
        """Memory used by layer."""
        return self.layer

    def mem_data(self, n):
        """Memory used by raw data."""
        return self.data


class Shapes3DSuite:
    """Benchmarks for the Shapes layer with 3D data."""

    params = [2**i for i in range(4, 9)]
    if "PR" in os.environ:
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
