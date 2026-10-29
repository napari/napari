from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt

from napari._qt.layer_controls.dynamic.widgets._points import (
    QtBorderColorControl,
    QtCurrentSizeSliderControl,
    QtSymbolComboBoxControl,
)
from napari.layers import Points

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtBorderColorControl:
    def test_init(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtBorderColorControl([points])
        qt_wrap.add_control(control)

    def test_border_color(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtBorderColorControl([points])
        qt_wrap.add_control(control)

        assert points.current_border_color == 'dimgrey'
        assert control.border_color_edit.line_edit.text() == 'dimgrey'

        points.current_border_color = 'white'
        assert control.border_color_edit.line_edit.text() == 'white'

        control.border_color_edit.line_edit.setFocus()
        control.border_color_edit.line_edit.setText('blue')
        qt_wrap.keyClick(control.border_color_edit.line_edit, Qt.Key.Key_Enter)
        assert points.current_border_color == 'blue'


class TestQtCurrentSizeSliderControl:
    def test_init(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtCurrentSizeSliderControl([points])
        qt_wrap.add_control(control)

    def test_size_change(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtCurrentSizeSliderControl([points])
        qt_wrap.add_control(control)

        assert points.current_size == 10
        assert control.size_slider.value() == 10

        control.size_slider.setValue(15)
        assert points.current_size == 15


class TestQtSymbolComboBoxControl:
    def test_init(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtSymbolComboBoxControl([points])
        qt_wrap.add_control(control)

    def test_symbol_change(self, qt_wrap: QtWrap):
        points = Points(data=[(0, 0), (10, 10)])
        control = QtSymbolComboBoxControl([points])
        qt_wrap.add_control(control)

        assert points.current_symbol == 'disc'
        assert control.symbol_combobox.currentText() == 'disc'

        control.symbol_combobox.setCurrentText('cross')
        assert points.current_symbol == 'cross'
