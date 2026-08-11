import numpy as np

from napari.components import Camera
from napari.components.cursor import Cursor


def test_cursor():
    """Test creating cursor object"""
    cursor = Cursor()
    assert cursor is not None


def test_cursor_view_direction():
    """View direction can be calculated from the cursor's canvas position."""
    cursor = Cursor()
    camera = Camera(
        center=(0, 0, 0), angles=(90, 0, 0), perspective=60, zoom=1
    )

    # view direction at the canvas center is the camera view direction
    view_direction = cursor.view_direction(
        camera=camera,
        canvas_size=(600, 600),
        canvas_position=(300, 300),
        ndim=3,
        dims_displayed=(0, 1, 2),
    )
    assert np.allclose(view_direction, camera.view_direction)

    # off-center position accounts for the field of view
    view_direction = cursor.view_direction(
        camera=camera,
        canvas_size=(600, 600),
        canvas_position=(450, 300),
        ndim=3,
        dims_displayed=(0, 1, 2),
    )
    assert np.allclose(view_direction, (-0.96076892, 0.27735010, 0), atol=1e-5)

    # fewer than three displayed dimensions returns None
    assert (
        cursor.view_direction(
            camera=camera,
            canvas_size=(600, 600),
            canvas_position=(300, 300),
            ndim=2,
            dims_displayed=(0, 1),
        )
        is None
    )
