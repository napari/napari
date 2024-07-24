from typing import TypeVar

import pandas as pd
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.qt_color_controls import (
    ConstantColorEncodingWidget,
    QuantitativeColorEncodingWidget,
)
from napari._tests.utils import assert_colors_equal
from napari.layers.utils.color_encoding import (
    ConstantColorEncoding,
    QuantitativeColorEncoding,
)

WidgetType = TypeVar('WidgetType', bound=QWidget)


def make_widget(qtbot: QtBot, widget_type: WidgetType) -> WidgetType:
    widget = widget_type()
    qtbot.addWidget(widget)
    return widget


def test_constant_color_encoding_widget_set_model(qtbot: QtBot):
    widget = make_widget(qtbot, ConstantColorEncodingWidget)
    encoding = ConstantColorEncoding(constant='blue')

    widget.setModel(encoding)

    assert widget.model is encoding
    assert_colors_equal(widget.constant.color, 'blue')


def test_constant_color_encoding_widget_set_model_constant(qtbot: QtBot):
    widget = make_widget(qtbot, ConstantColorEncodingWidget)
    encoding = ConstantColorEncoding(constant='blue')
    widget.setModel(encoding)

    encoding.constant = 'green'

    assert_colors_equal(widget.constant.color, 'green')


def test_constant_color_encoding_widget_set_widget_constant(qtbot: QtBot):
    widget = make_widget(qtbot, ConstantColorEncodingWidget)
    encoding = ConstantColorEncoding(constant='blue')
    widget.setModel(encoding)

    widget.constant.setColor('green')

    assert_colors_equal(encoding.constant, 'green')


def test_quantitative_color_encoding_widget_set_features(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )

    widget.setFeatures(features)

    widget_features = _combo_box_texts(widget.feature)
    assert widget_features == tuple(features.columns)
    assert widget.feature.currentText() == 'x'


def test_quantitative_color_encoding_widget_set_model(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    encoding = QuantitativeColorEncoding(feature='x', colormap='viridis')

    widget.setModel(encoding)

    assert widget.model is encoding
    assert widget.colormap.currentText() == encoding.colormap.name


def test_quantitative_color_encoding_widget_set_model_feature(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    encoding = QuantitativeColorEncoding(feature='x', colormap='viridis')
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )
    widget.setFeatures(features)
    widget.setModel(encoding)
    assert encoding.feature != 'y'

    encoding.feature = 'y'

    assert widget.feature.currentText() == 'y'


def test_quantitative_color_encoding_widget_set_widget_feature(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    encoding = QuantitativeColorEncoding(feature='x', colormap='viridis')
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )
    # TODO: maybe we should force features and model to be set together?
    # Otherwise, there are ordering issues (e.g. may want to only allow
    # setting a model that is compatible with features and vice versa?).
    widget.setFeatures(features)
    widget.setModel(encoding)
    assert widget.feature.currentText() != 'y'

    widget.feature.setCurrentText('y')

    # TODO: check if encoding state is also mutated, if desired.
    assert encoding.feature == 'y'


def test_quantitative_color_encoding_widget_set_model_colormap(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    encoding = QuantitativeColorEncoding(feature='x', colormap='viridis')
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )
    widget.setFeatures(features)
    widget.setModel(encoding)
    assert encoding.colormap.name != 'hsv'

    encoding.colormap = 'hsv'

    # TODO: consider asserting more than the name.
    assert widget.colormap.currentText() == 'hsv'

    # expected_values = encoding(features)
    # assert_colors_equal(encoding._values, expected_values)


def test_quantitative_color_encoding_widget_set_widget_colormap(qtbot: QtBot):
    widget = make_widget(qtbot, QuantitativeColorEncodingWidget)
    encoding = QuantitativeColorEncoding(feature='x', colormap='viridis')
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )
    widget.setFeatures(features)
    widget.setModel(encoding)
    assert widget.colormap.currentText() != 'hsv'

    widget.colormap.setCurrentText('hsv')

    # TODO: consider asserting more than the name.
    assert encoding.colormap.name == 'hsv'


def _combo_box_texts(combo_box: QComboBox) -> tuple[str, ...]:
    return tuple(combo_box.itemText(i) for i in range(combo_box.count()))
