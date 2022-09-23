import os

import numpy as np
import pytest

from napari import Viewer, layers
from napari._tests.utils import (
    add_layer_by_type,
    check_view_transform_consistency,
    check_viewer_functioning,
    layer_test_data,
    skip_local_popups,
    skip_on_win_ci,
)
from napari.utils._tests.test_naming import eval_with_filename
from napari.utils.action_manager import action_manager


def _get_all_keybinding_methods(type_):
    obj_methods = set(super(type_, type_).class_keymap.values())
    obj_methods.update({v.__name__ for v in type_.class_keymap.values()})
    obj_methods.update(
        {
            a.command.__name__
            for a in action_manager._get_layer_actions(type_).values()
        }
    )
    return obj_methods


viewer_methods = _get_all_keybinding_methods(Viewer)
EXPECTED_NUMBER_OF_VIEWER_METHODS = 14


def test_len_methods_viewer(make_napari_viewer):
    """
    Make sure we do find all the methods attached to a viewer via keybindings
    """
    _ = make_napari_viewer()
    viewer_methods = _get_all_keybinding_methods(Viewer)
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
    view = viewer.window._qt_viewer

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


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_add_layer(make_napari_viewer, layer_class, data, ndim):
    viewer = make_napari_viewer()
    layer = add_layer_by_type(viewer, layer_class, data, visible=True)
    check_viewer_functioning(viewer, viewer.window._qt_viewer, data, ndim)

    for func in layer.class_keymap.values():
        func(layer)


EXPECTED_NUMBER_OF_LAYER_METHODS = {
    'Image': 5,
    'Vectors': 0,
    'Surface': 0,
    'Tracks': 0,
    'Points': 9,
    'Labels': 14,
    'Shapes': 17,
}


@pytest.mark.parametrize(
    'cls, expectation', EXPECTED_NUMBER_OF_LAYER_METHODS.items()
)
def test_expected_number_of_layer_methods(cls, expectation):
    """
    Make sure we do find all the methods attached to a layer via keybindings
    """
    layer_methods = _get_all_keybinding_methods(getattr(layers, cls))
    assert len(layer_methods) == expectation


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


@skip_on_win_ci
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
    screenshot = viewer.screenshot(canvas_only=True, flash=False)
    assert screenshot.ndim == 3

    # Take screenshot with the viewer included
    screenshot = viewer.screenshot(canvas_only=False, flash=False)
    assert screenshot.ndim == 3


@skip_on_win_ci
def test_changing_theme(make_napari_viewer):
    """Test changing the theme updates the full window."""
    viewer = make_napari_viewer(show=False)
    viewer.window._qt_viewer.set_welcome_visible(False)
    viewer.add_points(data=None)
    size = viewer.window._qt_viewer.size()
    viewer.window._qt_viewer.setFixedSize(size)

    assert viewer.theme == 'dark'
    screenshot_dark = viewer.screenshot(canvas_only=False, flash=False)

    viewer.theme = 'light'
    assert viewer.theme == 'light'
    screenshot_light = viewer.screenshot(canvas_only=False, flash=False)

    equal = (screenshot_dark == screenshot_light).min(-1)

    # more than 99.5% of the pixels have changed
    assert (np.count_nonzero(equal) / equal.size) < 0.05, "Themes too similar"

    with pytest.raises(ValueError):
        viewer.theme = 'nonexistent_theme'


# TODO: revisit the need for sync_only here.
# An async failure was observed here on CI, but was not reproduced locally
@pytest.mark.sync_only
@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_roll_transpose_update(make_napari_viewer, layer_class, data, ndim):
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
    viewer.dims.transpose()
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


@skip_local_popups
def test_custom_layer(make_napari_viewer):
    """Make sure that custom layers subclasses can be added to the viewer."""

    class NewLabels(layers.Labels):
        """'Empty' extension of napari Labels layer."""

    # Make a viewer and add the custom layer
    viewer = make_napari_viewer(show=True)
    viewer.add_layer(NewLabels(np.zeros((10, 10, 10), dtype=np.uint8)))


def test_emitting_data_doesnt_change_points_value(make_napari_viewer):
    """Test emitting data with no change doesn't change the layer _value."""
    viewer = make_napari_viewer()

    data = np.array([[0, 0], [10, 10], [20, 20]])
    layer = viewer.add_points(data, size=2)
    viewer.layers.selection.active = layer

    assert layer._value is None
    viewer.mouse_over_canvas = True
    viewer.cursor.position = tuple(layer.data[1])
    assert layer._value == 1

    layer.events.data(value=layer.data)
    assert layer._value == 1


@pytest.mark.parametrize('layer_class, data, ndim', layer_test_data)
def test_emitting_data_doesnt_change_cursor_position(
    make_napari_viewer, layer_class, data, ndim
):
    """Test emitting data event from layer doesn't change cursor position"""
    viewer = make_napari_viewer()
    layer = layer_class(data)
    viewer.add_layer(layer)

    new_position = (5,) * ndim
    viewer.cursor.position = new_position
    layer.events.data(value=layer.data)

    assert viewer.cursor.position == new_position


@skip_local_popups
@skip_on_win_ci
def test_empty_shapes_dims(make_napari_viewer):
    """make sure an empty shapes layer can render in 3D"""
    viewer = make_napari_viewer(show=True)
    viewer.add_shapes(None)
    viewer.dims.ndisplay = 3


def test_current_viewer(make_napari_viewer):
    """Test that the viewer made last is the "current_viewer()" until another is activated"""
    # Make two DIFFERENT viewers
    viewer1: Viewer = make_napari_viewer()
    viewer2: Viewer = make_napari_viewer()
    assert viewer2 is not viewer1
    # Ensure one is returned by napari.current_viewer()
    from napari import current_viewer

    assert current_viewer() is viewer2
    assert current_viewer() is not viewer1

    viewer1.window.activate()

    assert current_viewer() is viewer1
    assert current_viewer() is not viewer2


def test_reset_empty(make_napari_viewer):
    """
    Test that resetting an empty viewer doesn't crash
    https://github.com/napari/napari/issues/4867
    """
    viewer = make_napari_viewer()
    viewer.reset()


def test_reset_non_empty(make_napari_viewer):
    """
    Test that resetting a non-empty viewer doesn't crash
    https://github.com/napari/napari/issues/4867
    """
    viewer = make_napari_viewer()
    viewer.add_points([(0, 1), (2, 3)])
    viewer.reset()
