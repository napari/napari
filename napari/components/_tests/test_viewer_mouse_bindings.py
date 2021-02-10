import numpy as np
import pytest

from napari.components import ViewerModel
from napari.components.cursor_event import CursorEvent
from napari.utils.interactions import ReadOnlyWrapper, mouse_wheel_callbacks


@pytest.mark.parametrize(
    "modifiers, inverted, expected_dim",
    [
        ([], True, [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]),
        (
            ["Control"],
            False,
            [[5, 5, 5], [5, 5, 4], [5, 5, 3], [5, 5, 0]],
        ),
        (
            ["Control"],
            True,
            [[5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 5, 9]],
        ),
    ],
)
def test_scroll(modifiers, inverted, expected_dim):
    """Test scrolling with mouse wheel."""
    viewer = ViewerModel()
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.dims.last_used = 2
    viewer.dims.set_point(axis=0, value=5)
    viewer.dims.set_point(axis=1, value=5)
    viewer.dims.set_point(axis=2, value=5)

    # Simulate tiny scroll
    event = ReadOnlyWrapper(
        CursorEvent(
            delta=[0, 0.6],
            modifiers=modifiers,
            inverted=inverted,
            type='mouse_wheel',
        )
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[0]).all()

    # Simulate tiny scroll
    event = ReadOnlyWrapper(
        CursorEvent(
            delta=[0, 0.6],
            modifiers=modifiers,
            inverted=inverted,
            type='mouse_wheel',
        )
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[1]).all()

    # Simulate tiny scroll
    event = ReadOnlyWrapper(
        CursorEvent(
            delta=[0, 0.9],
            modifiers=modifiers,
            inverted=inverted,
            type='mouse_wheel',
        )
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[2]).all()

    # Simulate large scroll
    event = ReadOnlyWrapper(
        CursorEvent(
            delta=[0, 3],
            modifiers=modifiers,
            inverted=inverted,
            type='mouse_wheel',
        )
    )
    mouse_wheel_callbacks(viewer, event)
    assert np.equal(viewer.dims.point, expected_dim[3]).all()
