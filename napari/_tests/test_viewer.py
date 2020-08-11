import os
import time

import numpy as np
import pytest

from napari import Viewer
from napari._tests.utils import (
    add_layer_by_type,
    check_view_transform_consistency,
    check_viewer_functioning,
    layer_test_data,
)


def test_viewer(make_test_viewer):
    """Test instantiating viewer."""
    viewer = make_test_viewer()
    view = viewer.window.qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.vbox_layout.count() == 2

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2

    # Run all class key bindings
    for func in viewer.class_keymap.values():
        # skip fullscreen test locally
        if func.__name__ == 'toggle_fullscreen' and not os.getenv("CI"):
            continue
        if func.__name__ == 'play':
            continue
        func(viewer)


@pytest.mark.run(order=1)  # provided by pytest-ordering
def test_no_qt_loop():
    """Test informative error raised when no Qt event loop exists.

    Logically, this test should go at the top of the file. Howveer, that
    resulted in tests passing when only this file was run, but failing when
    other tests involving Qt-bot were run before this file. Putting this test
    second provides a sanity check that pytest-ordering is correctly doing its
    magic.
    """
    with pytest.raises(RuntimeError):
        _ = Viewer()


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
@pytest.mark.parametrize('visible', [True, False])
def test_add_layer(make_test_viewer, layer_class, data, ndim, visible):
    viewer = make_test_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=visible)
    check_viewer_functioning(viewer, viewer.window.qt_viewer, data, ndim)

    # Run all class key bindings
    for func in layer.class_keymap.values():
        func(layer)


def test_screenshot(make_test_viewer):
    """Test taking a screenshot."""
    viewer = make_test_viewer()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Take screenshot of the image canvas only
    screenshot = viewer.screenshot(canvas_only=True)
    assert screenshot.ndim == 3

    # Take screenshot with the viewer included
    screenshot = viewer.screenshot(canvas_only=False)
    assert screenshot.ndim == 3


def test_update(make_test_viewer):
    data = np.random.random((512, 512))
    viewer = make_test_viewer()
    layer = viewer.add_image(data)

    def layer_update(*, update_period, num_updates):
        # number of times to update

        for k in range(num_updates):
            time.sleep(update_period)

            dat = np.random.random((512, 512))
            layer.data = dat

            assert layer.data.all() == dat.all()
            # if you're looking at this as an example,
            # it would be best to put a yield statement here...
            # but we're testing how it handles not having a yield statement

    # NOTE: The closure approach used here has the potential to throw an error:
    # "RuntimeError: Internal C++ object () already deleted."
    # if an enclosed object (like the layer here) is deleted in the main thread
    # and then subsequently called in the other thread.
    # Previously this error would have been invisible (raised only in the other
    # thread). But because this can make debugging hard, the new
    # `create_worker` approach reraises thread errors in the main thread by
    # default.  To make this test pass, we now need to explicitly use
    # `_ignore_errors=True`, because the `layer.data = dat` line will throw an
    # error when called after the main thread is closed.
    with pytest.warns(DeprecationWarning):
        viewer.update(
            layer_update,
            update_period=0.01,
            num_updates=100,
            _ignore_errors=True,
        )


def test_changing_theme(make_test_viewer):
    """Test changing the theme updates the full window."""
    viewer = make_test_viewer()
    viewer.add_points(data=None)
    assert viewer.palette['folder'] == 'dark'

    screenshot_dark = viewer.screenshot(canvas_only=False)

    viewer.theme = 'light'
    assert viewer.palette['folder'] == 'light'

    screenshot_light = viewer.screenshot(canvas_only=False)
    equal = (screenshot_dark == screenshot_light).min(-1)

    # more than 99.5% of the pixels have changed
    assert (np.count_nonzero(equal) / equal.size) < 0.05, "Themes too similar"

    with pytest.raises(ValueError):
        viewer.theme = 'nonexistent_theme'


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_roll_traspose_update(make_test_viewer, layer_class, data, ndim):
    """Check that transpose and roll preserve correct transform sequence."""

    viewer = make_test_viewer()

    np.random.seed(0)

    layer = add_layer_by_type(viewer, layer_class, data)

    # Set translations and scalings (match type of visual layer storing):
    transf_dict = {
        'translate': np.random.randint(0, 10, ndim).astype(np.float32),
        'scale': np.random.rand(ndim).astype(np.float32),
    }
    for k, val in transf_dict.items():
        setattr(layer, k, val)

    # Check consistency:
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Roll dims and check again:
    viewer.dims._roll()
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Transpose and check again:
    viewer.dims._transpose()
    check_view_transform_consistency(layer, viewer, transf_dict)
