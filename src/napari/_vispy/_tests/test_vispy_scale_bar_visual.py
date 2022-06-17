"""Test scale bar."""
import pytest
from pint import UndefinedUnitError

from napari.components._viewer_constants import Position


def test_vispy_text_visual(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window._qt_viewer
    assert viewer.scale_bar is not None
    assert qt_widget.scale_bar is not None

    # make sure units are not set yet
    assert qt_widget.scale_bar._unit_reg is None
    assert qt_widget.scale_bar._quantity is None

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
    for magnitude, unit, quantity in [
        (1, "", None),
        (1, "", ""),
        (12, "micrometer", "12um"),
        (13, "meter", "13 meters"),
        (0.5, "foot", "0.5ft"),
        (60, "second", "60s"),
    ]:
        viewer.scale_bar.unit = quantity
        assert qt_widget.scale_bar._quantity.magnitude == magnitude
        assert qt_widget.scale_bar._quantity.units == unit

    with pytest.raises(UndefinedUnitError):
        viewer.scale_bar.unit = "snail speed"

    # test to make sure unit is updated when scale bar is not visible
    viewer.scale_bar.visible = False
    viewer.scale_bar.unit = "pixel"
    assert qt_widget.scale_bar._quantity.units == "pixel"
    del qt_widget
