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

    def test_border_color(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeColorControl([shapes])
        qt_wrap.add_control(control)

        assert shapes.current_border_color == '#777777ff'
        assert control.border_color_edit.line_edit.text() == '#777777ff'

        shapes.current_border_color = 'red'
        assert control.border_color_edit.line_edit.text() == 'red'

        control.border_color_edit.line_edit.setFocus()
        control.border_color_edit.line_edit.setText('blue')
        qt_wrap.keyClick(control.border_color_edit.line_edit, Qt.Key.Key_Enter)
        assert shapes.current_border_color == 'blue'


class TestQtEdgeWidthSliderControl:
    def test_init(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeWidthSliderControl([shapes])
        qt_wrap.add_control(control)

    def test_border_width(self, qt_wrap: QtWrap):
        shapes = Shapes()
        control = QtEdgeWidthSliderControl([shapes])
        qt_wrap.add_control(control)

        assert shapes.current_border_width == 1.0
        assert control.border_width_slider.value() == 1

        shapes.current_border_width = 5.0
        assert control.border_width_slider.value() == 5

        control.border_width_slider.setValue(10)
        assert shapes.current_border_width == 10.0
