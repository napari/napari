"""For testing one off action callbacks"""
from napari._app_model.actions._view_actions import _tooltip_visibility_toggle
from napari.settings import get_settings


def test_tooltip_visibility_toggle():
    settings = get_settings().appearance
    assert settings.layer_tooltip_visibility is False
    _tooltip_visibility_toggle()
    assert settings.layer_tooltip_visibility is True
    _tooltip_visibility_toggle()
    assert settings.layer_tooltip_visibility is False
