from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt

from napari._qt.layer_controls.dynamic.widgets._shapes import (
    QtEdgeColorControl,
    QtEdgeWidthSliderControl,
)
from napari.layers import Shapes

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtEdgeColorControl:
    def test_init(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeColorControl([shapes])
        qt_wrap.add_control(control)

    def test_edge_color(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeColorControl([shapes])
        qt_wrap.add_control(control)

        assert shapes.current_edge_color == '#777777ff'
        assert control.edge_color_edit.line_edit.text() == '#777777ff'

        shapes.current_edge_color = 'red'
        assert control.edge_color_edit.line_edit.text() == 'red'

        control.edge_color_edit.line_edit.setFocus()
        control.edge_color_edit.line_edit.setText('blue')
        qt_wrap.keyClick(control.edge_color_edit.line_edit, Qt.Key.Key_Enter)
        assert shapes.current_edge_color == 'blue'


class TestQtEdgeWidthSliderControl:
    def test_init(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeWidthSliderControl([shapes])
        qt_wrap.add_control(control)

    def test_edge_width(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeWidthSliderControl([shapes])
        qt_wrap.add_control(control)

        assert shapes.current_edge_width == 1.0
        assert control.edge_width_slider.value() == 1

        shapes.current_edge_width = 5.0
        assert control.edge_width_slider.value() == 5

        control.edge_width_slider.setValue(10)
        assert shapes.current_edge_width == 10.0
