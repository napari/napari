from typing import TypeVar

import pandas as pd
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.qt_color_controls import (
    ColorMode,
    ColorModeWidget,
)
from napari.layers import Vectors
from napari.layers.utils.color_encoding import (
    ConstantColorEncoding,
    ManualColorEncoding,
    QuantitativeColorEncoding,
)

WidgetType = TypeVar('WidgetType', bound=QWidget)


def make_widget(qtbot: QtBot, widget_type: WidgetType, *args) -> WidgetType:
    widget = widget_type(*args)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def widget(qtbot: QtBot) -> ColorModeWidget:
    data = [
        [[0, 0], [1, 1]],
        [[1, 1], [2, 2]],
        [[2, 2], [3, 3]],
    ]
    features = pd.DataFrame(
        {
            'x': [0, 0.5, 1],
            'y': [False, True, True],
        }
    )
    layer = Vectors(data, features=features)
    return make_widget(qtbot, ColorModeWidget, layer, 'edge_color')


def test_color_mode_widget_init(widget: ColorModeWidget):
    assert_widget_mode(widget, ColorMode.MANUAL)
    style_attr = getattr(widget.layer.style, widget.attr)
    assert widget._manual.model is style_attr


def test_color_mode_widget_set_mode_manual(widget: ColorModeWidget):
    # Instead of this, create a layer with a constant color encoding.
    widget.mode.setCurrentText('constant')
    widget.mode.setCurrentText('manual')
    assert_widget_mode(widget, ColorMode.MANUAL)
    style_attr = getattr(widget.layer.style, widget.attr)
    assert widget._manual.model is style_attr


def test_color_mode_widget_set_attr_manual(widget: ColorModeWidget):
    widget.mode.setCurrentText('constant')
    encoding = ManualColorEncoding(array=[], default='blue')
    setattr(widget.layer.style, widget.attr, encoding)
    assert_widget_mode(widget, ColorMode.MANUAL)
    assert widget._manual.model is encoding


def test_color_mode_widget_set_mode_constant(widget: ColorModeWidget):
    widget.mode.setCurrentText('constant')
    assert_widget_mode(widget, ColorMode.CONSTANT)
    style_attr = getattr(widget.layer.style, widget.attr)
    assert widget._constant.model is style_attr


def test_color_mode_widget_set_attr_constant(widget: ColorModeWidget):
    encoding = ConstantColorEncoding(constant='blue')
    setattr(widget.layer.style, widget.attr, encoding)
    assert_widget_mode(widget, ColorMode.CONSTANT)
    assert widget._constant.model is encoding


def test_color_mode_widget_set_mode_quantitative(widget: ColorModeWidget):
    widget.mode.setCurrentText('quantitative')
    assert_widget_mode(widget, ColorMode.QUANTITATIVE)
    style_attr = getattr(widget.layer.style, widget.attr)
    assert widget._quantitative.model is style_attr


def test_color_mode_widget_set_attr_quantitative(widget: ColorModeWidget):
    encoding = QuantitativeColorEncoding(
        feature='x',
        colormap='viridis',
    )
    setattr(widget.layer.style, widget.attr, encoding)
    assert_widget_mode(widget, ColorMode.QUANTITATIVE)
    assert widget._quantitative.model is encoding


def assert_widget_mode(widget: ColorModeWidget, mode: ColorMode) -> None:
    current_mode = widget.mode.currentText()
    assert current_mode == mode
    encoding_widget = widget.encodings[ColorMode(mode)]
    assert encoding_widget.isVisibleTo(widget)
    assert all(
        not w.isVisibleTo(widget)
        for w in widget.encodings.values()
        if w is not encoding_widget
    )
