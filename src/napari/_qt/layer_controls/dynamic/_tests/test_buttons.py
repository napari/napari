from __future__ import annotations

from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pytest
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QMessageBox, QWidget

from napari._qt.layer_controls.dynamic.buttons import (
    QtImageButtons,
    QtLabelsButtons,
    QtMultiLayerButtons,
    QtPointsButtons,
    QtShapesButtons,
    QtSurfaceButtons,
    QtTracksButtons,
    QtVectorsButtons,
)
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)


@pytest.fixture
def widget(qtbot) -> QWidget:
    """Fixture to create a QWidget for testing."""
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


class TestQtImageButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Image(data=np.zeros((10, 10), dtype=np.uint8))
        buttons = QtImageButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

    def test_invalid_mode(self, widget: QWidget) -> None:
        layer = Image(data=np.zeros((10, 10), dtype=np.uint8))
        buttons = QtImageButtons(layer)
        widget.setLayout(buttons)

        with pytest.raises(
            ValueError, match="Mode 'invalid_mode' not recognized"
        ):
            buttons._on_mode_change(Namespace(mode='invalid_mode'))

    def test_transform_reset(
        self, widget: QWidget, monkeypatch: pytest.MonkeyPatch, qapp
    ) -> None:
        layer = Image(data=np.zeros((10, 10), dtype=np.uint8))
        reset_mock = Mock()
        warning_mock = Mock(return_value=QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(layer, '_reset_affine', reset_mock)
        monkeypatch.setattr(QMessageBox, 'warning', warning_mock)

        buttons = QtImageButtons(layer)
        widget.setLayout(buttons)

        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(buttons.transform_button.rect().center()),
            QPointF(buttons.transform_button.rect().center()),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
        )

        qapp.sendEvent(buttons.transform_button, event)

        reset_mock.assert_called_once()
        warning_mock.assert_called_once()


class TestQtLabelsButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        buttons = QtLabelsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

    def test_mode_switching(self, widget: QWidget) -> None:
        layer = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        buttons = QtLabelsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        layer.mode = 'paint'
        assert buttons.paint_button.isChecked()

        layer.mode = 'erase'
        assert buttons.paint_button.isChecked() is False
        assert buttons.erase_button.isChecked()

    def test_new_colormap(self, widget: QWidget) -> None:
        layer = Labels(data=np.zeros((10, 10), dtype=np.uint8))
        buttons = QtLabelsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        cmap = layer.colormap
        buttons.change_color()
        assert layer.colormap is not cmap


class TestQtPointsButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Points(data=[])
        buttons = QtPointsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

    def test_mode_switching(self, widget: QWidget) -> None:
        layer = Points(data=[])
        buttons = QtPointsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        layer.mode = 'add'
        assert buttons.addition_button.isChecked()

        layer.mode = 'select'
        assert buttons.addition_button.isChecked() is False
        assert buttons.select_button.isChecked()

    def test_ndisplay_change(self, widget: QWidget) -> None:
        layer = Points(data=[])
        buttons = QtPointsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        # Change ndisplay to 3D
        buttons.ndisplay = 3
        assert layer.editable is False

        # Change ndisplay back to 2D
        buttons.ndisplay = 2
        assert layer.editable is True


class TestQtShapesButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Shapes(data=[])
        buttons = QtShapesButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

    def test_mode_switching(self, widget: QWidget) -> None:
        layer = Shapes(data=[])
        buttons = QtShapesButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        layer.mode = 'add_rectangle'
        assert buttons.rectangle_button.isChecked()

        layer.mode = 'select'
        assert buttons.rectangle_button.isChecked() is False
        assert buttons.select_button.isChecked()

    def test_ndisplay_change(self, widget: QWidget) -> None:
        layer = Shapes(data=[])
        buttons = QtShapesButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer

        # Change ndisplay to 3D
        buttons.ndisplay = 3
        assert layer.editable is False

        # Change ndisplay back to 2D
        buttons.ndisplay = 2
        assert layer.editable is True


class TestQtSurfaceButtons:
    def test_init(self, widget: QWidget, surface_data) -> None:
        layer = Surface(data=surface_data)
        buttons = QtSurfaceButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer


class TestQtTracksButtons:
    def test_init(self, widget: QWidget, tracks_data) -> None:
        layer = Tracks(
            data=tracks_data['data'], properties=tracks_data['properties']
        )
        buttons = QtTracksButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer


class TestQtVectorsButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Vectors(data=[])
        buttons = QtVectorsButtons(layer)
        widget.setLayout(buttons)
        assert buttons.layer == layer


class TestQtMultiLayerButtons:
    def test_init(self, widget: QWidget) -> None:
        layer = Shapes(data=[])
        buttons = QtMultiLayerButtons(layer)
        widget.setLayout(buttons)
        assert buttons.count() == 1, (
            'QtMultiLayerButtons should have Pan Zoom button'
        )
