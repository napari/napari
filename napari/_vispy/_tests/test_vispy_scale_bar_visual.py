"""Test scale bar"""
import pytest
from pint import UndefinedUnitError

from napari.components._viewer_constants import Position


def test_vispy_text_visual(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window.qt_viewer
    assert viewer.scale_bar is not None
    assert qt_widget.scale_bar is not None

    # check visible attribute
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible
    viewer.scale_bar.visible = True
    assert (
        qt_widget.scale_bar.node.visible
        == qt_widget.scale_bar.text_node.visible
        == viewer.scale_bar.visible
        is True
    )

    # check font size attribute
    assert (
        qt_widget.scale_bar.text_node.font_size == viewer.scale_bar.font_size
    )
    viewer.scale_bar.font_size = 13
    assert (
        qt_widget.scale_bar.text_node.font_size
        == viewer.scale_bar.font_size
        == 13
    )

    # check ticks attribute
    viewer.scale_bar.ticks = False
    assert len(qt_widget.scale_bar.node._pos) == 2
    viewer.scale_bar.ticks = True
    assert len(qt_widget.scale_bar.node._pos) == 6

    # check visible attribute
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible
    viewer.scale_bar.visible = True
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible

    # check position attribute
    for position in list(Position):
        viewer.scale_bar.position = position
        assert viewer.scale_bar.position == position
    with pytest.raises(ValueError):
        viewer.scale_bar.position = "top_centre"

    # check a couple of pint's units
    for magnitude, unit in [
        (1, ""),
        (12, "12um"),
        (13, "13 meters"),
        (0.5, "0.5ft"),
        (60, "60s"),
    ]:
        viewer.scale_bar.unit = unit
        assert qt_widget.scale_bar._quantity.magnitude == magnitude

    with pytest.raises(UndefinedUnitError):
        viewer.scale_bar.unit = "snail speed"
