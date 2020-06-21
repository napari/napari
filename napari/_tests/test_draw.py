import numpy as np
import pytest
import sys


@pytest.mark.skipif(
    sys.platform.startswith('win') or sys.platform.startswith('linux'),
    reason='Currently fails on certain CI due to error on canvas draw.',
)
def test_canvas_drawing(viewer_factory):
    """Test drawing before and after adding and then deleting a layer."""
    view, viewer = viewer_factory()
    assert len(viewer.layers) == 0
    # Check canvas context is not none before drawing, as currently on
    # some of our CI a proper canvas context is not made
    view.canvas.events.draw()

    # Add layer
    data = np.random.random((15, 10, 5))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    view.canvas.events.draw()

    # Remove layer
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    view.canvas.events.draw()
