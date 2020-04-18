import numpy as np


def test_canvas_drawing(viewer_factory):
    """Test drawing before and after adding and then deleting a layer."""
    view, viewer = viewer_factory()
    assert len(viewer.layers) == 0
    view.canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    # Check canvas context is not none before drawing, as currently on
    # some of our CI a proper canvas context is not made
    if view.canvas.context is not None:
        view.canvas.events.draw()

    # Add layer
    data = np.random.random((15, 10, 5))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    if view.canvas.context is not None:
        view.canvas.events.draw()

    # Remove layer
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    if view.canvas.context is not None:
        view.canvas.events.draw()
