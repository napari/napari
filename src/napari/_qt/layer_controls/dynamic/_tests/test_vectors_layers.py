from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from napari._qt.layer_controls.dynamic.widgets._vectors import (
    QtEdgeColorFeatureControl,
    QtLengthSpinBoxControl,
    QtVectorStyleComboBoxControl,
    QtWidthSpinBoxControl,
)
from napari.layers import Vectors

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtEdgeColorFeatureControl:
    def test_init(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

    def test_edge_color_mode(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(
            vectors_data, properties={'feature': [1, 2], 'feature2': [3, 4]}
        )
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.edge_color_mode == 'direct'
        assert control.color_mode_combobox.currentText() == 'direct'

        with pytest.warns(
            RuntimeWarning, match='edge_color property was not set'
        ):
            vectors.edge_color_mode = 'cycle'
        assert control.color_mode_combobox.currentText() == 'cycle'

        control.color_mode_combobox.setCurrentText('direct')
        assert vectors.edge_color_mode == 'direct'

    def test_change_edge_color_feature(
        self, qt_wrap: QtWrap, vectors_data
    ) -> None:
        vectors = Vectors(
            vectors_data,
            properties={'feature': [1, 2], 'feature2': ['a', 'b']},
        )
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

        assert control.color_feature_box.count() == 2
        assert control.color_feature_box.currentText() == 'feature'

        control.color_feature_box.setCurrentText('feature2')
        assert vectors.edge_color_mode == 'cycle'

    def test_colormap_controls_follow_the_mode(
        self, qt_wrap: QtWrap, vectors_data
    ) -> None:
        vectors = Vectors(vectors_data)
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

        assert control.colormap_combobox.isHidden()
        assert control.contrast_limits_slider.isHidden()

        vectors.features = {'phase': np.array([-np.pi, np.pi])}
        vectors.edge_color = 'phase'
        vectors.edge_color_mode = 'colormap'
        assert not control.colormap_combobox.isHidden()
        assert not control.contrast_limits_slider.isHidden()

        # widget -> layer
        index = control.colormap_combobox.findData('twilight')
        control.colormap_combobox.setCurrentIndex(index)
        control.contrast_limits_slider.setValue((-1.0, 1.0))
        assert vectors.edge_colormap.name == 'twilight'
        np.testing.assert_allclose(vectors.edge_contrast_limits, (-1.0, 1.0))

        # layer -> widget
        vectors.edge_colormap = 'hsv'
        vectors.edge_contrast_limits = (-2.0, 2.0)
        assert control.colormap_combobox.currentData() == 'hsv'
        np.testing.assert_allclose(
            control.contrast_limits_slider.value(), (-2.0, 2.0)
        )


class TestQtWidthSpinBoxControl:
    def test_init(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtWidthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

    def test_change_width(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtWidthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.edge_width == 1.0
        control.width_spinbox.setValue(2.5)
        assert vectors.edge_width == 2.5

        vectors.edge_width = 3.0
        assert control.width_spinbox.value() == 3.0


class TestQtLengthSpinBoxControl:
    def test_init(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtLengthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

    def test_change_length(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtLengthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.length == 1.0
        control.length_spinbox.setValue(2.5)
        assert vectors.length == 2.5

        vectors.length = 3.0
        assert control.length_spinbox.value() == 3.0


class TestQtVectorStyleComboBoxControl:
    def test_init(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtVectorStyleComboBoxControl([vectors])
        qt_wrap.add_control(control)

    def test_change_vector_style(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtVectorStyleComboBoxControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.vector_style == 'triangle'
        assert control.vector_style_combobox.currentText() == 'triangle'

        control.vector_style_combobox.setCurrentText('arrow')
        assert vectors.vector_style == 'arrow'

        vectors.vector_style = 'triangle'
        assert control.vector_style_combobox.currentText() == 'triangle'
