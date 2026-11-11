from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np

from napari._qt.layer_controls.dynamic.widgets._labels import (
    QtBrushSizeSliderControl,
    QtColorModeComboBoxControl,
    QtContiguousCheckBoxControl,
    QtContourSpinBoxControl,
    QtCurrentLabelControls,
    QtDisplaySelectedLabelCheckBoxControl,
    QtLabelRenderingControl,
    QtNdimSpinBoxControl,
    QtPreserveLabelsCheckBoxControl,
)
from napari.layers import Labels
from napari.layers.labels._labels_constants import LabelColorMode
from napari.utils import CyclicLabelColormap, DirectLabelColormap

if TYPE_CHECKING:
    import pytest

    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtBrushSizeSliderControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtBrushSizeSliderControl([labels])
        qt_wrap.add_control(control)

    def test_brush_size_update(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtBrushSizeSliderControl([labels])
        qt_wrap.add_control(control)

        assert labels.brush_size == 10
        assert control.brush_size_slider.value() == 10

        labels.brush_size = 20
        assert control.brush_size_slider.value() == 20

        control.brush_size_slider.setValue(30)
        assert labels.brush_size == 30


class TestQtColorModeComboBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtColorModeComboBoxControl([labels])
        qt_wrap.add_control(control)

    def test_update_colormap(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtColorModeComboBoxControl([labels])
        qt_wrap.add_control(control)

        labels.colormap = {
            0: 'black',
            1: 'blue',
            2: 'red',
            3: 'yellow',
            None: 'white',
        }

        assert (
            control.color_mode_combobox.currentEnum() == LabelColorMode.DIRECT
        )
        assert isinstance(labels.colormap, DirectLabelColormap)

        control.color_mode_combobox.setCurrentEnum(LabelColorMode.AUTO)
        control.change_color_mode()  # We need to trigger this manually as callback is triggered ony on user interaction.

        assert isinstance(labels.colormap, CyclicLabelColormap)

        control.color_mode_combobox.setCurrentEnum(LabelColorMode.DIRECT)
        control.change_color_mode()  # We need to trigger this manually as callback is triggered ony on user interaction.

        assert isinstance(labels.colormap, DirectLabelColormap)


class TestQtContiguousCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtContiguousCheckBoxControl([labels])
        qt_wrap.add_control(control)

    def test_contiguous_update(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtContiguousCheckBoxControl([labels])
        qt_wrap.add_control(control)

        assert labels.contiguous is True
        assert control.contiguous_checkbox.isChecked() is True

        labels.contiguous = False
        assert control.contiguous_checkbox.isChecked() is False

        control.contiguous_checkbox.setChecked(True)
        assert labels.contiguous is True


class TestQtContourSpinBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtContourSpinBoxControl([labels])
        qt_wrap.add_control(control)

    def test_contour_update(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtContourSpinBoxControl([labels])
        qt_wrap.add_control(control)

        assert labels.contour == 0
        assert control.contour_spinbox.value() == 0

        labels.contour = 5
        assert control.contour_spinbox.value() == 5

        control.contour_spinbox.setValue(10)
        assert labels.contour == 10


class TestQtCurrentLabelControls:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtCurrentLabelControls([labels])
        qt_wrap.add_control(control)

    def test_selected_label_update(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        labels.data[0, 0] = 1
        control = QtCurrentLabelControls([labels])
        qt_wrap.add_control(control)

        assert labels.selected_label == 1
        assert control.selection_spinbox.value() == 1

        control.new_label_button.click()
        assert labels.selected_label == 2
        assert control.selection_spinbox.value() == 2

        labels.selected_label = 5
        assert control.selection_spinbox.value() == 5

        control.selection_spinbox.setValue(10)
        assert labels.selected_label == 10

    def test_update_data_range(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtCurrentLabelControls([labels])
        qt_wrap.add_control(control)

        assert control.selection_spinbox.minimum() == 0
        assert control.selection_spinbox.maximum() == 255

        labels.data = np.zeros((10, 10), dtype=np.uint16)
        assert control.selection_spinbox.minimum() == 0
        assert control.selection_spinbox.maximum() == 65535

    def test_colorbox_update(
        self, qt_wrap: QtWrap, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = MagicMock()
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtCurrentLabelControls([labels])
        qt_wrap.add_control(control)

        monkeypatch.setattr(
            'napari._qt.layer_controls.dynamic.widgets._labels.qt_current_label_controls.QPainter',
            mock,
        )

        assert labels.selected_label == 1
        control.colorbox.paintEvent(None)
        mock.return_value.drawRect.assert_called_once()  # Check that the colorbox is being painted

        mock.reset_mock()
        labels.selected_label = 0
        control.colorbox.paintEvent(None)
        assert mock.return_value.drawRect.call_count == 36


class TestQtDisplaySelectedLabelCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDisplaySelectedLabelCheckBoxControl([labels])
        qt_wrap.add_control(control)

    def test_show_selected_sync(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDisplaySelectedLabelCheckBoxControl([labels])
        qt_wrap.add_control(control)

        assert not labels.show_selected_label
        assert not control.selected_color_checkbox.isChecked()

        labels.show_selected_label = True
        assert control.selected_color_checkbox.isChecked()

        control.selected_color_checkbox.setChecked(False)
        assert not labels.show_selected_label


class TestQtNdimSpinBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtNdimSpinBoxControl([labels])
        qt_wrap.add_control(control)

    def test_change_ndim_edit(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((5, 10, 10), dtype=np.uint8))
        control = QtNdimSpinBoxControl([labels])
        qt_wrap.add_control(control)

        assert control.ndim_spinbox.value() == 2
        assert labels.n_edit_dimensions == 2

        control.ndim_spinbox.setValue(3)
        assert labels.n_edit_dimensions == 3

        labels.n_edit_dimensions = 2
        assert control.ndim_spinbox.value() == 2

    def test_change_max_value(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((5, 10, 10), dtype=np.uint8))
        control = QtNdimSpinBoxControl([labels])
        qt_wrap.add_control(control)

        assert control.ndim_spinbox.maximum() == 3

        labels.data = np.zeros((5, 10, 10, 10), dtype=np.uint8)
        assert control.ndim_spinbox.maximum() == 4


class TestQtPreserveLabelsCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtPreserveLabelsCheckBoxControl([labels])
        qt_wrap.add_control(control)

    def test_preserve_labels_sync(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtPreserveLabelsCheckBoxControl([labels])
        qt_wrap.add_control(control)

        assert not labels.preserve_labels
        assert (
            control.preserve_labels_checkbox.isChecked()
            == labels.preserve_labels
        )

        labels.preserve_labels = not labels.preserve_labels
        assert (
            control.preserve_labels_checkbox.isChecked()
            == labels.preserve_labels
        )

        control.preserve_labels_checkbox.setChecked(
            not control.preserve_labels_checkbox.isChecked()
        )
        assert (
            control.preserve_labels_checkbox.isChecked()
            == labels.preserve_labels
        )


class TestQtLabelRenderControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtLabelRenderingControl([labels])
        qt_wrap.add_control(control)

    def test_rendering_sync(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtLabelRenderingControl([labels])
        qt_wrap.add_control(control)

        assert labels.rendering == 'iso_categorical'
        assert control.rendering_combobox.currentText() == labels.rendering

        labels.rendering = 'translucent'
        assert control.rendering_combobox.currentText() == 'translucent'

        control.rendering_combobox.setCurrentText('iso_categorical')
        assert labels.rendering == 'iso_categorical'

    def test_iso_gradient_mode(self, qt_wrap: QtWrap) -> None:
        labels = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtLabelRenderingControl([labels])
        qt_wrap.add_control(control)

        assert labels.iso_gradient_mode == 'fast'
        assert control.iso_gradient_combobox.currentText() == 'fast'

        labels.iso_gradient_mode = 'smooth'
        assert control.iso_gradient_combobox.currentText() == 'smooth'

        control.iso_gradient_combobox.setCurrentText('fast')
        assert labels.iso_gradient_mode == 'fast'
