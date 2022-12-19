"""Test scale bar."""
import numpy as np
import pytest
from pint import UndefinedUnitError

from napari.components import ViewerModel
from napari.components._viewer_constants import CanvasPosition
from napari.qt import QtViewer


def test_vispy_text_visual(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_widget = viewer.window._qt_viewer
    assert viewer.scale_bar is not None
    assert qt_widget.scale_bar is not None

    # check visible attribute
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible
    viewer.scale_bar.visible = True
    assert qt_widget.scale_bar.node.visible

    # check font size attribute
    assert (
        qt_widget.scale_bar.node.text.font_size == viewer.scale_bar.font_size
    )
    viewer.scale_bar.font_size = 13
    assert qt_widget.scale_bar.node.text.font_size == 13

    # check ticks attribute
    viewer.scale_bar.ticks = False
    assert len(qt_widget.scale_bar.node.line._pos) == 2
    viewer.scale_bar.ticks = True
    assert len(qt_widget.scale_bar.node.line._pos) == 6

    # check visible attribute
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible
    viewer.scale_bar.visible = True
    assert qt_widget.scale_bar.node.visible == viewer.scale_bar.visible

    # check position attribute
    for position in list(CanvasPosition):
        viewer.scale_bar.position = position
        assert viewer.scale_bar.position == position

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
        assert qt_widget.scale_bar._unit.magnitude == magnitude
        assert qt_widget.scale_bar._unit.units == unit

    with pytest.raises(UndefinedUnitError):
        viewer.scale_bar.unit = "snail speed"

    # test to make sure unit is updated when scale bar is not visible
    viewer.scale_bar.visible = False
    viewer.scale_bar.unit = "pixel"
    assert qt_widget.scale_bar._unit.units == "pixel"

    # test box visible attribute
    assert qt_widget.scale_bar.node.box.visible == viewer.scale_bar.box
    viewer.scale_bar.box = True
    assert viewer.scale_bar.box

    # test color attributes
    viewer.scale_bar.colored = True
    for (rgba, color) in [
        ((0.0, 1.0, 1.0, 1.0), "#00ffff"),  # check hex color
        ((1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),  # check 3 tuple
        ((1.0, 0.5, 0.0, 0.5), (1.0, 0.5, 0.0, 0.5)),  # check 4 tuple
        ((1.0, 1.0, 1.0, 1.0), "white"),  # check text color
    ]:
        viewer.scale_bar.color = color
        np.testing.assert_equal(viewer.scale_bar.color, np.asarray(rgba))

    viewer.scale_bar.box = True
    for (rgba, color) in [
        ((0.0, 1.0, 1.0, 1.0), "#00ffff"),  # check hex color
        ((1.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),  # check 3 tuple
        ((1.0, 0.5, 0.0, 0.5), (1.0, 0.5, 0.0, 0.5)),  # check 4 tuple
        ((1.0, 1.0, 1.0, 1.0), "white"),  # check text color
    ]:
        viewer.scale_bar.box_color = color
        np.testing.assert_equal(viewer.scale_bar.box_color, np.asarray(rgba))

    del qt_widget


def test_box_color_with_background_color_override(qtbot):
    viewer_model = ViewerModel()
    qt_viewer = QtViewer(viewer_model)
    qtbot.addWidget(qt_viewer)

    qt_viewer.canvas.background_color_override = [1, 0, 0]
    qt_viewer.scale_bar.reset()

    np.testing.assert_equal(
        qt_viewer.scale_bar.node.text.color.rgb, [[0, 1, 1]]
    )

    qt_viewer.canvas.background_color_override = [0, 1, 0]
    qt_viewer.scale_bar.reset()

    np.testing.assert_equal(
        qt_viewer.scale_bar.node.text.color.rgb, [[1, 0, 1]]
    )
