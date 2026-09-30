import numpy as np
import pytest

from napari._vispy.overlays.text import (
    VispyLayerNameOverlay,
    _relative_luminance,
)
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.layers import Image, Points, Surface
from napari.utils.colormaps import Colormap

pytestmark = pytest.mark.usefixtures('qapp')


def _make_overlay(layer, background='black'):
    viewer = ViewerModel()
    viewer.canvas.background_color_override = background
    return VispyLayerNameOverlay(
        layer=layer,
        overlay=layer.name_overlay,
        viewer=viewer,
        font_info=FontInfo(),
    )


def test_name_overlay_uses_colormap_color():
    layer = Image(np.zeros((2, 2)), colormap='green')
    vispy_overlay = _make_overlay(layer)
    np.testing.assert_allclose(vispy_overlay.node.color.rgba[0], [0, 1, 0, 1])

    layer.colormap = 'magenta'
    np.testing.assert_allclose(vispy_overlay.node.color.rgba[0], [1, 0, 1, 1])


def test_name_overlay_explicit_color_wins():
    layer = Image(np.zeros((2, 2)), colormap='green')
    vispy_overlay = _make_overlay(layer)
    layer.name_overlay.color = 'red'
    np.testing.assert_allclose(vispy_overlay.node.color.rgba[0], [1, 0, 0, 1])


@pytest.mark.parametrize(
    'make_layer',
    [
        lambda: Image(np.zeros((2, 2)), colormap='gray'),
        lambda: Image(np.zeros((2, 2)), colormap='gray_r'),
        Points,
    ],
    ids=['gray', 'gray_r', 'points'],
)
def test_name_overlay_falls_back_to_contrasting_color(make_layer):
    vispy_overlay = _make_overlay(make_layer())
    np.testing.assert_allclose(
        vispy_overlay.node.color.rgba[0], vispy_overlay._get_fgcolor()
    )


def _contrast(a, b):
    lighter, darker = sorted(
        (_relative_luminance(a[:3]), _relative_luminance(b[:3])), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize(
    ('cmap', 'background', 'channel'),
    [('green', 'white', 1), ('cyan', 'white', 2), ('blue', 'black', 2)],
)
def test_name_overlay_keeps_hue_but_stays_readable(cmap, background, channel):
    vispy_overlay = _make_overlay(
        Image(np.zeros((2, 2)), colormap=cmap), background
    )
    color = vispy_overlay.node.color.rgba[0]
    bg = vispy_overlay._get_bgcolor()
    assert _contrast(color, bg) >= 3
    assert color[channel] == color[:3].max()
    assert np.ptp(color[:3]) > 0.3


def test_name_overlay_surface_and_transparent_colormap():
    surface = Surface(
        (np.eye(3), np.array([[0, 1, 2]]), np.ones(3)), colormap='red'
    )
    np.testing.assert_allclose(
        _make_overlay(surface).node.color.rgba[0], [1, 0, 0, 1]
    )

    fading = Colormap([[0, 0, 0, 0], [1, 0, 0, 0]])
    image = Image(np.zeros((2, 2)), colormap=fading)
    assert _make_overlay(image).node.color.rgba[0][3] == 1
