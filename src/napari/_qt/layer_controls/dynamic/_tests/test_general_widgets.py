from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._qt.layer_controls.dynamic.widgets import (
    QtColormapControl,
    QtContrastLimitsControl,
    QtFaceColorControl,
    QtGammaSliderControl,
    QtHistogramControl,
)
from napari.layers import Image, Points

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestColormapControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

    def test_change_colormap(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

        assert image.colormap.name == 'gray'
        assert control.colormap_combobox.currentText() == 'gray'

        control.colormap_combobox.setCurrentText('magma')
        assert image.colormap.name == 'magma'


class TestQtContrastLimitsControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

    def test_change_contrast_limits(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        assert np.all(image.contrast_limits == [0, 255])
        assert np.all(control.contrast_limits_slider.value() == (0, 255))

        control.contrast_limits_slider.setValue((50, 200))
        assert np.all(image.contrast_limits == [50, 200])


class TestQtFaceColorControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtFaceColorControl([points])
        qt_wrap.add_control(control)

    def test_change_face_color(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtFaceColorControl([points])
        qt_wrap.add_control(control)

        assert points.current_face_color == 'white'  # Default white
        assert np.all(control.face_color_edit.color == (1, 1, 1, 1))

        control.face_color_edit.setColor((1, 0, 0))  # Set to red
        assert points.current_face_color == 'red'

        points.current_face_color = 'blue'
        assert np.all(control.face_color_edit.color == (0, 0, 1, 1))


class TestQtGammaSliderControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtGammaSliderControl([image])
        qt_wrap.add_control(control)

    def test_update_gamma(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtGammaSliderControl([image])
        qt_wrap.add_control(control)

        assert image.gamma == 1.0
        assert control.gamma_slider.value() == 1.0

        control.gamma_slider.setValue(2.0)
        assert image.gamma == 2.0

        image.gamma = 0.5
        assert control.gamma_slider.value() == 0.5


class TestQtHistogramControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtHistogramControl([image])
        qt_wrap.add_control(control)
        qt_wrap.add_widget(control.content_widget)

    def test_histogram_update(self, qt_wrap: QtWrap) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtHistogramControl([image])
        qt_wrap.add_control(control)
        qt_wrap.add_widget(control.content_widget)
        control.ensure_content()
