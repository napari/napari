import pytest

from napari.components._viewer_key_bindings import (
    hold_for_pan_zoom,
    toggle_theme,
)
from napari.components.viewer_model import ViewerModel
from napari.layers.points import Points
from napari.settings import get_settings
from napari.utils.theme import available_themes, get_system_theme


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
