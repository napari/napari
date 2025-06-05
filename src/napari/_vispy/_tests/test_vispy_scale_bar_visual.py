from unittest.mock import MagicMock

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
    assert scale_bar.node.box.height == 36
    assert scale_bar.y_offset == 7

    model.position = 'top_right'
    # y_offset should be increase to account for box height
    assert scale_bar.y_offset == 7 + scale_bar.node.box.height / 2

    # increasing font while at top increases y_offset due to new box height
    model.font_size = 30
    assert scale_bar.node.box.height == 63
    assert scale_bar.y_size == 63 / 2
    assert scale_bar.y_offset == 7 + scale_bar.y_size

    # moving scale bar back to bottom should reset y_offset to 7
    # y_size and box height should remain the same
    model.position = 'bottom_right'
    assert scale_bar.y_offset == 7
    assert scale_bar.node.box.height == 63
    assert scale_bar.y_size == 63 / 2

    # changing font_size at bottom should have no effect on the offset
    # but should increase the box height and y_size
    model.font_size = 40
    assert scale_bar.y_offset == 7
    assert scale_bar.y_size == 38
    assert scale_bar.node.box.height == 76

    # setting `colored` should have no effect
    model.colored = True
    assert scale_bar.y_offset == 7
    assert scale_bar.y_size == 38
    assert scale_bar.node.box.height == 76
    # check that the line is not at the center,
    # but offset down due to the font size (40)
    assert scale_bar.node.line.pos[0, 1] == scale_bar.node.box.height / 2 - 18

    # changing `ticks` should have no effect
    model.ticks = False
    assert scale_bar.y_offset == 7
    assert scale_bar.y_size == 38
    assert scale_bar.node.box.height == 76
    # check that the line is not at the center,
    # but offset down due to the font size (40)
    assert scale_bar.node.line.pos[0, 1] == scale_bar.node.box.height / 2 - 18
