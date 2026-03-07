import numpy as np
import pytest

from napari._vispy.visuals.welcome import Welcome


@pytest.mark.usefixtures('qapp')
def test_welcome_uses_rasterized_text():
    welcome = Welcome()
    welcome.set_color(np.array([1, 1, 1, 1], dtype=float))
    welcome.set_version('0.0.0')
    welcome.set_shortcuts(())
    welcome.set_tip('Shortcut symbols: ⇧⌘⌥⌃↵')
    welcome.set_scale_and_position(1200, 800)

    texture = welcome.text_image._data
    assert texture is not None
    assert texture.ndim == 3
    assert texture.shape[-1] == 4
    assert texture[..., 3].max() > 0


@pytest.mark.usefixtures('qapp')
def test_welcome_text_and_logo_transforms_scale_with_canvas():
    welcome = Welcome()
    welcome.set_version('0.0.0')
    welcome.set_tip('Did you know this is a rasterized text texture?')
    welcome.set_scale_and_position(1000, 600)

    expected_scale = min(1000, 600) * 0.002
    np.testing.assert_allclose(
        welcome.logo.transform.scale[:2], (expected_scale, expected_scale)
    )
    np.testing.assert_allclose(
        welcome.text_image.transform.scale[:2],
        (
            expected_scale / welcome._TEXT_RASTER_SCALE,
            expected_scale / welcome._TEXT_RASTER_SCALE,
        ),
    )
