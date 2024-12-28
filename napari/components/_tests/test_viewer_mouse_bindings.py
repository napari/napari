from unittest.mock import Mock

import numpy as np
import pytest

from napari.components import ViewerModel
from napari.components._viewer_mouse_bindings import double_click_to_zoom
from napari.utils._test_utils import read_only_mouse_event
from napari.utils.interactions import mouse_wheel_callbacks


class WheelEvent:
    def __init__(self, inverted) -> None:
        self._inverted = inverted

    def inverted(self):
        return self._inverted


@pytest.mark.parametrize(
    ('modifiers', 'native', 'expected_dim'),
    [
        ([], WheelEvent(True), [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]),
        (
            ['Control'],
            WheelEvent(False),
            [[5, 5, 5], [4, 5, 5], [3, 5, 5], [0, 5, 5]],
        ),
        (
            ['Control'],
            WheelEvent(True),
            [[5, 5, 5], [6, 5, 5], [7, 5, 5], [9, 5, 5]],
        ),
    ],
)
def test_paint(modifiers, native, expected_dim):
    """Test painting labels with circle/square brush."""
    viewer = ViewerModel()
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.dims.last_used = 0
    viewer.dims.set_point(axis=0, value=5)
    viewer.dims.set_point(axis=1, value=5)
    viewer.dims.set_point(axis=2, value=5)

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.6], modifiers=modifiers, native=native, type='wheel'
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[0]).all()

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.6], modifiers=modifiers, native=native, type='wheel'
    )

    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[1]).all()

    # Simulate tiny scroll
    event = read_only_mouse_event(
        delta=[0, 0.9], modifiers=modifiers, native=native, type='wheel'
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[2]).all()

    # Simulate large scroll
    event = read_only_mouse_event(
        delta=[0, 3], modifiers=modifiers, native=native, type='wheel'
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[3]).all()


def test_double_click_to_zoom():
    viewer = ViewerModel()
    data = np.zeros((10, 10, 10))
    viewer.add_image(data)

    # Ensure `pan_zoom` mode is active
    assert viewer.layers.selection.active.mode == 'pan_zoom'

    # Mock the mouse event
    event = Mock()
    event.modifiers = []
    event.position = [100, 100]

    viewer.camera.center = (0, 0, 0)
    initial_zoom = viewer.camera.zoom
    initial_center = np.asarray(viewer.camera.center)
    assert viewer.dims.ndisplay == 2

    double_click_to_zoom(viewer, event)

    assert viewer.camera.zoom == initial_zoom * 2
    # should be half way between the old center and the event.position
    assert np.allclose(viewer.camera.center, (0, 50, 50))

    # Assert the camera center has moved correctly in 3D
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    # reset to initial values
    viewer.camera.center = initial_center
    viewer.camera.zoom = initial_zoom

    event.position = [0, 100, 100]
    double_click_to_zoom(viewer, event)
    assert viewer.camera.zoom == initial_zoom * 2
    assert np.allclose(viewer.camera.center, (0, 50, 50))

    # Test with Alt key pressed
    event.modifiers = ['Alt']

    double_click_to_zoom(viewer, event)

    # Assert the zoom level is back to initial
    assert viewer.camera.zoom == initial_zoom
    # Assert the camera center is back to initial
    assert np.allclose(viewer.camera.center, (0, 0, 0))

    # Test in a mode other than pan_zoom
    viewer.layers.selection.active.mode = 'transform'
    assert viewer.layers.selection.active.mode != 'pan_zoom'

    double_click_to_zoom(viewer, event)

    # Assert nothing has changed
    assert viewer.camera.zoom == initial_zoom
    assert np.allclose(viewer.camera.center, (0, 0, 0))
