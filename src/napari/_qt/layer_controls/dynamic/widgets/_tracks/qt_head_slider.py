from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSlider,
    QWidget,
)

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Tracks


class QtHeadLengthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current head length
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Tracks]
        A list of napari Tracks layers.

    Attributes
    ----------
    head_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling head length of the layer.
    head_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the head length chooser widget.
    """

    _layers: list[Tracks]

    def __init__(
        self, layers: list[Tracks], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.head_length.connect(self._on_head_length_change)

        # Setup widgets
        # slider for track head length
        self.head_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.head_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.head_length_slider.setMinimum(0)
        self.head_length_slider.setMaximum(self._layers[0]._max_length)
        self.head_length_slider.setSingleStep(1)
        for layer in self._layers:
            connect_setattr(
                self.head_length_slider.valueChanged, layer, 'head_length'
            )
        self.head_length_slider_label = QtWrappedLabel('head length:')

    def _on_head_length_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.head_length_slider):
            value = self._layers[0].head_length
            if value > self.head_length_slider.maximum():
                self.head_length_slider.setMaximum(self._layers[0]._max_length)
            self.head_length_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QWidget, ...]]:
        return [(self.head_length_slider_label, self.head_length_slider)]
