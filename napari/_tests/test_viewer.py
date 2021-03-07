import os

import numpy as np
import pytest

from napari import Viewer, layers
from napari._tests.utils import (
    add_layer_by_type,
    check_view_transform_consistency,
    check_viewer_functioning,
    layer_test_data,
)
from napari.utils._tests.test_naming import eval_with_filename

viewer_methods = set(super(Viewer, Viewer).class_keymap.values())
viewer_methods.update(Viewer.class_keymap.values())


def test_len_methods_viewer():
    assert len(viewer_methods) == 18


@pytest.mark.parametrize('func', viewer_methods)
def test_viewer(make_napari_viewer, func):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
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
    if func.__name__ == 'toggle_fullscreen' and not os.getenv("CI"):
        return
    if func.__name__ == 'play':
        return
    func(viewer)


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
@pytest.mark.parametrize('visible', [True, False])
def test_add_layer(make_napari_viewer, layer_class, data, ndim, visible):
    viewer = make_napari_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=visible)
    check_viewer_functioning(viewer, viewer.window.qt_viewer, data, ndim)

    methods = set(super(type(layer), layer).class_keymap.values())
    methods.update(layer.class_keymap.values())
    # Run all class key bindings
    assert len(methods) == 13
    for func in methods:
        func(layer)


@pytest.mark.parametrize('layer_class, a_unique_name, ndim', layer_test_data)
def test_add_layer_magic_name(
    make_napari_viewer, layer_class, a_unique_name, ndim
):
    """Test magic_name works when using add_* for layers"""
    # Tests for issue #1709
    viewer = make_napari_viewer()  # noqa: F841
    layer = eval_with_filename(
        "add_layer_by_type(viewer, layer_class, a_unique_name)",
        "somefile.py",
    )
    assert layer.name == "a_unique_name"


def test_screenshot(make_napari_viewer):
    """Test taking a screenshot."""
    viewer = make_napari_viewer()

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


def test_changing_theme(make_napari_viewer):
    """Test changing the theme updates the full window."""
    viewer = make_napari_viewer()
    viewer.add_points(data=None)
    assert viewer.theme == 'dark'

    screenshot_dark = viewer.screenshot(canvas_only=False)

    viewer.theme = 'light'
    assert viewer.theme == 'light'

    screenshot_light = viewer.screenshot(canvas_only=False)
    equal = (screenshot_dark == screenshot_light).min(-1)

    # more than 99.5% of the pixels have changed
    assert (np.count_nonzero(equal) / equal.size) < 0.05, "Themes too similar"

    with pytest.raises(ValueError):
        viewer.theme = 'nonexistent_theme'


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_roll_traspose_update(make_napari_viewer, layer_class, data, ndim):
    """Check that transpose and roll preserve correct transform sequence."""

    viewer = make_napari_viewer()

    np.random.seed(0)

    layer = add_layer_by_type(viewer, layer_class, data)

    # Set translations and scalings (match type of visual layer storing):
    transf_dict = {
        'translate': np.random.randint(0, 10, ndim).astype(np.float32),
        'scale': np.random.rand(ndim).astype(np.float32),
    }
    for k, val in transf_dict.items():
        setattr(layer, k, val)

    if layer_class in [layers.Image, layers.Labels]:
        transf_dict['translate'] -= transf_dict['scale'] / 2

    # Check consistency:
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Roll dims and check again:
    viewer.dims._roll()
    check_view_transform_consistency(layer, viewer, transf_dict)

    # Transpose and check again:
    viewer.dims._transpose()
    check_view_transform_consistency(layer, viewer, transf_dict)


def test_toggling_axes(make_napari_viewer):
    """Test toggling axes."""
    viewer = make_napari_viewer()

    # Check axes are not visible
    assert not viewer.axes.visible

    # Make axes visible
    viewer.axes.visible = True
    assert viewer.axes.visible

    # Enter 3D rendering and check axes still visible
    viewer.dims.ndisplay = 3
    assert viewer.axes.visible

    # Make axes not visible
    viewer.axes.visible = False
    assert not viewer.axes.visible


def test_toggling_scale_bar(make_napari_viewer):
    """Test toggling scale bar."""
    viewer = make_napari_viewer()

    # Check scale bar is not visible
    assert not viewer.scale_bar.visible

    # Make scale bar visible
    viewer.scale_bar.visible = True
    assert viewer.scale_bar.visible

    # Enter 3D rendering and check scale bar is still visible
    viewer.dims.ndisplay = 3
    assert viewer.scale_bar.visible

    # Make scale bar not visible
    viewer.scale_bar.visible = False
    assert not viewer.scale_bar.visible
