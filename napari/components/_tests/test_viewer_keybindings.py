from napari.components._viewer_key_bindings import toggle_theme
from napari.settings import get_settings


def test_theme_toggle_keybinding(make_napari_viewer):
    viewer = make_napari_viewer()
    current_theme = get_settings().appearance.theme
    assert current_theme == 'dark'
    toggle_theme(viewer)
    # toggle_theme should not change settings
    assert not current_theme == 'light'
    # toggle_theme should change the current theme
    assert viewer.theme == 'light'
    toggle_theme(viewer)
    # toggle_theme should toggle between themes
    # currently only dark and light themes are available
    assert viewer.theme == 'dark'
