from napari.components._viewer_key_bindings import toggle_theme
from napari.settings import get_settings


def test_theme_toggle_keybinding(make_napari_viewer):
    viewer = make_napari_viewer()
    current_theme = get_settings().appearance.theme
    assert viewer.theme == current_theme
    toggle_theme(viewer)
    # toggle_theme should not change settings
    assert not current_theme == 'light'
    # toggle_theme should change the viewer theme
    assert viewer.theme == 'light'
    toggle_theme(viewer)
    # toggle_theme should toggle between actual themes
    assert not viewer.theme == 'system'
