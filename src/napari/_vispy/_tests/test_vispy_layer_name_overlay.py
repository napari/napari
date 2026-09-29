import numpy as np
import pytest

from napari._vispy.overlays.text import VispyLayerNameOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.layers import Image, Points

pytestmark = pytest.mark.usefixtures('qapp')


def _make_overlay(layer):
    return VispyLayerNameOverlay(
        layer=layer,
        overlay=layer.name_overlay,
        viewer=ViewerModel(),
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
