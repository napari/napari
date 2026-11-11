from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.dynamic.widgets._surface import (
    QtShadingComboBoxControl,
)
from napari.layers import Surface

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtShadingComboBoxControl:
    def test_init(self, qt_wrap: QtWrap, surface_data) -> None:
        surface = Surface(surface_data)
        control = QtShadingComboBoxControl([surface])
        qt_wrap.add_control(control)

    def test_shading(self, qt_wrap: QtWrap, surface_data) -> None:
        surface = Surface(surface_data)
        control = QtShadingComboBoxControl([surface])
        qt_wrap.add_control(control)

        assert surface.shading == 'flat'
        assert control.shading_combobox.currentText() == 'flat'

        surface.shading = 'smooth'
        assert control.shading_combobox.currentText() == 'smooth'

        control.shading_combobox.setCurrentText('flat')
        assert surface.shading == 'flat'
