from unittest.mock import MagicMock

import pytest

from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
from napari.components.overlays import ScaleBarOverlay


def test_scale_bar_instantiation(make_napari_viewer):
    viewer = make_napari_viewer()
    model = ScaleBarOverlay()
    vispy_scale_bar = VispyScaleBarOverlay(overlay=model, viewer=viewer)
    assert vispy_scale_bar.overlay.length is None
    model.length = 50
    assert vispy_scale_bar.overlay.length == 50


def test_scale_bar_positioning(make_napari_viewer):
    viewer = make_napari_viewer()
    # set devicePixelRatio to 2 so testing works on CI and local
    viewer.window._qt_window.devicePixelRatio = MagicMock(return_value=2)
    model = ScaleBarOverlay()
    scale_bar = VispyScaleBarOverlay(overlay=model, viewer=viewer)

    assert model.position == 'bottom_right'
    assert model.font_size == 10
    assert scale_bar.y_offset == 20

    model.position = 'top_right'
    assert scale_bar.y_offset == pytest.approx(20.333, abs=0.1)

    # increasing size while at top should increase y_offset to 7 + font_size*1.33
    model.font_size = 30
    assert scale_bar.y_offset == pytest.approx(47, abs=0.1)

    # moving scale bar back to bottom should reset y_offset to 20
    model.position = 'bottom_right'
    assert scale_bar.y_offset == 20

    # changing font_size at bottom should have no effect
    model.font_size = 50
    assert scale_bar.y_offset == 20
