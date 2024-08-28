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

    # This cannot work unless the encoding keeps a reference to features,
    # because it needs that to generate new values.
    expected_values = encoding(features)
    # We need to explicitly call apply instead.
    # This makes me think that create a new encoding instance every time
    # the widget changes might be the better approach.
    # Though in this case, it is the encoding itself that is being mutated.
    # To fix that, it either needs to be immutable (to mirror what the widget
    # does), or it needs to store a weak reference to features and update
    # appropriately.
    # Alternatively, the widget could store a weak reference to features
    # (especially since it needs the update features for the combobox)
    # but that feels a little weird because the encoding will behave
    # differently based on whether its connected to the widget (which is
    # true right now). Or we could recreate the encoding instance on the
    # widget and mutate it on the layer, though that might be surprising
    # for the user (since they may have some original instance).
    # TODO: maybe it's enough to have the widget listen to features
    # and encoding and regenerate the cached values as needed.
    encoding._apply(features)
    assert_colors_equal(encoding._values, expected_values)


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
