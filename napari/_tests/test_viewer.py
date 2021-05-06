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


def _get_all_keybinding_methods(type_):
    obj_methods = set(super(type_, type_).class_keymap.values())
    obj_methods.update(type_.class_keymap.values())
    return obj_methods


viewer_methods = _get_all_keybinding_methods(Viewer)
EXPECTED_NUMBER_OF_VIEWER_METHODS = 19


def test_len_methods_viewer():
    """
    Make sure we do find all the methods attached to a viewer via keybindings
    """
    assert len(viewer_methods) == EXPECTED_NUMBER_OF_VIEWER_METHODS


@pytest.mark.xfail
def test_non_existing_bindings():
    """
    Those are condition tested in next unittest; but do not exists; this is
    likely due to an oversight somewhere.
    """
    assert 'play' in [x.__name__ for x in viewer_methods]
    assert 'toggle_fullscreen' in [x.__name__ for x in viewer_methods]


@pytest.mark.parametrize('func', viewer_methods)
def test_viewer_methods(make_napari_viewer, func):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()

    if func.__name__ == 'toggle_fullscreen' and not os.getenv("CI"):
        pytest.skip("Fullscreen cannot be tested in CI")
    if func.__name__ == 'play':
        pytest.skip("Play cannot be tested with Pytest")
    func(viewer)


def test_viewer(make_napari_viewer):
    """Test instantiating viewer."""
    viewer = make_napari_viewer()
    view = viewer.window.qt_viewer

    assert viewer.title == 'napari'
    assert view.viewer == viewer

    assert len(viewer.layers) == 0
    assert view.layers.model().rowCount() == 0

    assert viewer.dims.ndim == 2
    assert view.dims.nsliders == viewer.dims.ndim
    assert np.sum(view.dims._displayed_sliders) == 0

    # Switch to 3D rendering mode and back to 2D rendering mode
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    viewer.dims.ndisplay = 2
    assert viewer.dims.ndisplay == 2


EXPECTED_NUMBER_OF_LAYER_METHODS = {
    'Image': 0,
    'Vectors': 0,
    'Surface': 0,
    'Tracks': 0,
    'Points': 8,
    'Labels': 14,
    'Shapes': 17,
}


# We unroll the layer data, with the all the methods of the layer that we are
# going to test, so that if one method fails we know which one, as well as
# remove potential issues that would be triggered by calling methods after each
# other.


unrolled_layer_data = []
for layer_class, data, ndim in layer_test_data:
    methods = _get_all_keybinding_methods(layer_class)
    for func in methods:
        unrolled_layer_data.append(
            (layer_class, data, ndim, func, len(methods))
        )


@pytest.mark.parametrize(
    'layer_class, data, ndim, func, Nmeth', unrolled_layer_data
)
@pytest.mark.parametrize('visible', [True, False])
def test_add_layer(
    make_napari_viewer, layer_class, data, ndim, func, Nmeth, visible
):
    viewer = make_napari_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=visible)
    check_viewer_functioning(viewer, viewer.window.qt_viewer, data, ndim)

    func(layer)

    assert Nmeth == EXPECTED_NUMBER_OF_LAYER_METHODS[layer_class.__name__]


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
    viewer = make_napari_viewer(show=False)
    viewer.window.qt_viewer.set_welcome_visible(False)
    viewer.add_points(data=None)
    size = viewer.window.qt_viewer.size()
    viewer.window.qt_viewer.setFixedSize(size)

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


def test_removing_points_data(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.random((4, 2)) * 4

    pts_layer = viewer.add_points(points)
    pts_layer.data = np.zeros([0, 2])

    assert len(pts_layer.data) == 0


def test_deleting_points(make_napari_viewer):
    viewer = make_napari_viewer()
    points = np.random.random((4, 2)) * 4

    pts_layer = viewer.add_points(points)
    pts_layer.selected_data = {0}
    pts_layer.remove_selected()

    assert len(pts_layer.data) == 3
