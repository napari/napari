from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

from napari._qt._qapp_model.injection._qprocessors import (
    _add_future_data,
    _add_layer_data_to_viewer,
    _add_layer_data_tuples_to_viewer,
    _add_layer_to_viewer,
    _add_plugin_dock_widget,
)
from napari.types import ImageData, LabelsData


def test_add_plugin_dock_widget(qtbot):
    widget = QWidget()
    viewer = MagicMock()
    qtbot.addWidget(widget)
    with pytest.raises(RuntimeError, match='No current `Viewer` found.'):
        _add_plugin_dock_widget((widget, 'widget'))
    _add_plugin_dock_widget((widget, 'widget'), viewer)


def test_add_layer_data_tuples_to_viewer():
    viewer = MagicMock()
    error_data = (np.zeros((10, 10)), np.zeros((10, 10)))
    valid_data = [(np.zeros((10, 10)),), (np.zeros((10, 10)),)]
    with pytest.raises(
        TypeError, match='Not a valid list of layer data tuples!'
    ):
        _add_layer_data_tuples_to_viewer(
            data=error_data,
            return_type=Union[ImageData, LabelsData],
            viewer=viewer,
        )
    _add_layer_data_tuples_to_viewer(
        data=valid_data,
        return_type=Union[ImageData, LabelsData],
        viewer=viewer,
    )


def test_add_layer_data_to_viewer():
    v = MagicMock()
    with pytest.raises(TypeError, match='napari supports only Optional'):
        _add_layer_data_to_viewer(
            data=np.zeros((10, 10)),
            return_type=Union[ImageData, LabelsData],
            viewer=v,
        )


def test_add_layer_to_viewer():
    layer = MagicMock()
    viewer = MagicMock()
    _add_layer_to_viewer(layer)
    _add_layer_to_viewer(layer, viewer)


def test_add_future_data():
    future = MagicMock()
    _add_future_data(future, Union[ImageData, LabelsData])
