import numpy as np
import pytest

from napari._tests.utils import (
    add_layer_by_type,
    layer_test_data,
)
from napari.components._viewer_key_bindings import (
    hold_for_pan_zoom,
    rotate_layers,
    show_only_layer_above,
    show_only_layer_below,
    toggle_selected_visibility,
    toggle_theme,
    toggle_unselected_visibility,
)
from napari.components.viewer_model import ViewerModel
from napari.layers.points import Points
from napari.settings import get_settings
from napari.utils.theme import available_themes, get_system_theme


@pytest.mark.key_bindings
def test_theme_toggle_keybinding():
    viewer = ViewerModel()
    assert viewer.theme == get_settings().appearance.theme
    assert viewer.theme != 'light'
    toggle_theme(viewer)
    # toggle_theme should not change settings
    assert get_settings().appearance.theme != 'light'
    # toggle_theme should change the viewer theme
    assert viewer.theme == 'light'
    # ensure toggle_theme loops through all themes
    initial_theme = viewer.theme
    number_of_actual_themes = len(available_themes())
    if 'system' in available_themes():
        number_of_actual_themes = len(available_themes()) - 1
    for _i in range(number_of_actual_themes):
        current_theme = viewer.theme
        toggle_theme(viewer)
        # theme should have changed
        assert viewer.theme != current_theme
        # toggle_theme should toggle only actual themes
        assert viewer.theme != 'system'
    # ensure we're back at the initial theme
    assert viewer.theme == initial_theme


def test_theme_toggle_from_system_theme():
    get_settings().appearance.theme = 'system'
    viewer = ViewerModel()
    assert viewer.theme == 'system'
    actual_initial_theme = get_system_theme()
    toggle_theme(viewer)
    # ensure that theme has changed
    assert viewer.theme != actual_initial_theme
    assert viewer.theme != 'system'
    number_of_actual_themes = len(available_themes())
    if 'system' in available_themes():
        number_of_actual_themes = len(available_themes()) - 1
    for _i in range(number_of_actual_themes - 1):  # we've already toggled once
        current_theme = viewer.theme
        toggle_theme(viewer)
        # theme should have changed
        assert viewer.theme != current_theme
        # toggle_theme should toggle only actual themes
        assert viewer.theme != 'system'
    # ensure we have looped back to whatever system was
    assert viewer.theme == actual_initial_theme


def test_hold_for_pan_zoom():
    viewer = ViewerModel()
    data = [[1, 3], [8, 4], [10, 10], [15, 4]]
    layer = Points(data, size=1)
    viewer.layers.append(layer)
    layer.mode = 'transform'

    viewer.layers.selection.active = viewer.layers[0]
    gen = hold_for_pan_zoom(viewer)
    assert layer.mode == 'transform'
    next(gen)
    assert layer.mode == 'pan_zoom'
    with pytest.raises(StopIteration):
        next(gen)
    assert layer.mode == 'transform'


def test_selected_visibility_toggle():
    viewer = make_viewer_with_three_layers()
    viewer.layers.selection.active = viewer.layers[0]
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible
    toggle_selected_visibility(viewer)
    assert not viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible
    toggle_selected_visibility(viewer)
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible


def test_unselected_visibility_toggle():
    viewer = make_viewer_with_three_layers()
    viewer.layers.selection.active = viewer.layers[0]
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible
    toggle_unselected_visibility(viewer)
    assert viewer.layers[0].visible
    assert not viewer.layers[1].visible
    assert not viewer.layers[2].visible
    toggle_unselected_visibility(viewer)
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible


def test_show_only_layer_above():
    viewer = make_viewer_with_three_layers()
    viewer.layers.selection.active = viewer.layers[0]
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible
    show_only_layer_above(viewer)
    assert not viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert not viewer.layers[2].visible
    show_only_layer_above(viewer)
    assert not viewer.layers[0].visible
    assert not viewer.layers[1].visible
    assert viewer.layers[2].visible


def test_show_only_layer_below():
    viewer = make_viewer_with_three_layers()
    viewer.layers.selection.active = viewer.layers[2]
    assert viewer.layers[0].visible
    assert viewer.layers[1].visible
    assert viewer.layers[2].visible
    show_only_layer_below(viewer)
    assert not viewer.layers[2].visible
    assert viewer.layers[1].visible
    assert not viewer.layers[0].visible
    show_only_layer_below(viewer)
    assert not viewer.layers[2].visible
    assert not viewer.layers[1].visible
    assert viewer.layers[0].visible


@pytest.mark.parametrize(('layer_class', 'data', 'ndim'), layer_test_data)
def test_rotate_layers(layer_class, data, ndim):
    """Test rotate layers works with all layer types/data"""
    viewer = ViewerModel()
    layer = add_layer_by_type(viewer, layer_class, data, visible=True)
    np.testing.assert_array_equal(
        layer.affine.rotate, np.eye(ndim, dtype=float)
    )
    rotate_layers(viewer)
    np.testing.assert_array_equal(
        layer.affine.rotate[-2:, -2:], np.array([[0, -1], [1, 0]], dtype=float)
    )


def test_rotate_layers_in_3D():
    """Test that rotate layers is disabled in 3D viewer mode"""
    viewer = ViewerModel()
    layer = add_layer_by_type(
        viewer, Points, np.array([[1, 2, 3]]), visible=True
    )
    initial_rotation_matrix = layer.affine.rotate[-2:, -2:]
    viewer.dims.ndisplay = 3
    assert viewer.dims.ndisplay == 3
    rotate_layers(viewer)
    # with ndisplay == 3 rotation is disabled, the matrix should not have changed
    np.testing.assert_array_equal(
        layer.affine.rotate[-2:, -2:], initial_rotation_matrix
    )


def make_viewer_with_three_layers():
    """Helper function to create a viewer with three layers"""
    viewer = ViewerModel()
    layer1 = Points()
    layer2 = Points()
    layer3 = Points()
    viewer.layers.append(layer1)
    viewer.layers.append(layer2)
    viewer.layers.append(layer3)
    return viewer
