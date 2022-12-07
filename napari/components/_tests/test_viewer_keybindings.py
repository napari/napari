from napari.components._viewer_key_bindings import toggle_theme
from napari.components.viewer_model import ViewerModel
from napari.settings import get_settings
from napari.utils.theme import available_themes, get_system_theme


def test_theme_toggle_keybinding():
    viewer = ViewerModel()
    assert viewer.theme == get_settings().appearance.theme
    assert not viewer.theme == 'light'
    toggle_theme(viewer)
    # toggle_theme should not change settings
    assert not get_settings().appearance.theme == 'light'
    # toggle_theme should change the viewer theme
    assert viewer.theme == 'light'
    # ensure toggle_theme loops through all themes
    initial_theme = viewer.theme
    number_of_actual_themes = len(available_themes())
    if 'system' in available_themes():
        number_of_actual_themes = len(available_themes()) - 1
    for i in range(number_of_actual_themes):
        current_theme = viewer.theme
        toggle_theme(viewer)
        # theme should have changed
        assert not viewer.theme == current_theme
        # toggle_theme should toggle only actual themes
        assert not viewer.theme == 'system'
    # ensure we're back at the initial theme
    assert viewer.theme == initial_theme


def test_theme_toggle_from_system_theme():
    get_settings().appearance.theme = 'system'
    viewer = ViewerModel()
    actual_theme = get_system_theme()
    toggle_theme(viewer)
    # ensure that theme has changed
    assert not viewer.theme == actual_theme
    toggle_theme(viewer)
    # ensure we have looped back to whatever system was
    assert viewer.theme == actual_theme
