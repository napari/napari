import sys

import numpy as np
import pytest

from napari._tests.utils import skip_local_popups


@skip_local_popups
@pytest.mark.skipif(
    sys.platform.startswith('win') or sys.platform.startswith('linux'),
    reason='Currently fails on certain CI due to error on canvas draw.',
)
def test_canvas_drawing(make_napari_viewer):
    """Test drawing before and after adding and then deleting a layer."""
    viewer = make_napari_viewer(show=True)
    view = viewer.window._qt_viewer
    view.set_welcome_visible(False)

    assert len(viewer.layers) == 0

    # Check canvas context is not none before drawing, as currently on
    # some of our CI a proper canvas context is not made
    view.canvas.scene_canvas.events.draw()

    # Add layer
    data = np.random.random((15, 10, 5))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    view.canvas.scene_canvas.events.draw()

    # Remove layer
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    view.canvas.scene_canvas.events.draw()
