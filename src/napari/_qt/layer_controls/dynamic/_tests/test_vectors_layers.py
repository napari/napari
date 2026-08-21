from __future__ import annotations

from typing import TYPE_CHECKING

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

    def test_color_mode(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(
            vectors_data, properties={'feature': [1, 2], 'feature2': [3, 4]}
        )
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.color_mode == 'direct'
        assert control.color_mode_combobox.currentText() == 'direct'

        with pytest.warns(RuntimeWarning, match='color property was not set'):
            vectors.color_mode = 'cycle'
        assert control.color_mode_combobox.currentText() == 'cycle'

        control.color_mode_combobox.setCurrentText('direct')
        assert vectors.color_mode == 'direct'

    def test_change_color_feature(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(
            vectors_data,
            properties={'feature': [1, 2], 'feature2': ['a', 'b']},
        )
        control = QtEdgeColorFeatureControl([vectors])
        qt_wrap.add_control(control)

        assert control.color_feature_box.count() == 2
        assert control.color_feature_box.currentText() == 'feature'

        control.color_feature_box.setCurrentText('feature2')
        assert vectors.color_mode == 'cycle'


class TestQtWidthSpinBoxControl:
    def test_init(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtWidthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

    def test_change_width(self, qt_wrap: QtWrap, vectors_data) -> None:
        vectors = Vectors(vectors_data)
        control = QtWidthSpinBoxControl([vectors])
        qt_wrap.add_control(control)

        assert vectors.width == 1.0
        control.width_spinbox.setValue(2.5)
        assert vectors.width == 2.5

        vectors.width = 3.0
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
