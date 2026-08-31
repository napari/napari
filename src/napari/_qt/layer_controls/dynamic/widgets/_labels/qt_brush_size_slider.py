from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from superqt import QLabeledSlider

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Labels


class QtBrushSizeSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current brush
    size attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Labels]
        A list of napari Labels layers.

    Attributes
    ----------
    brush_size_slider : superqt.QLabeledDoubleSlider
        Slider controlling current brush size of the layer.
    brush_size_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the brush size chooser widget.
    """

    _layers: list[Labels]

    def __init__(
        self, layers: list[Labels], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.brush_size.connect(self._on_brush_size_change)

        # Setup widgets
        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        for layer in self._layers:
            connect_setattr(sld.valueChanged, layer, 'brush_size')
        self.brush_size_slider = sld
        self._on_brush_size_change()

        self.brush_size_slider_label = QtWrappedLabel('brush size:')

    def _on_brush_size_change(self) -> None:
        """Receive layer model brush size change event and update the slider."""
        with qt_signals_blocked(self.brush_size_slider):
            value = self._layers[0].brush_size
            value = np.maximum(1, int(value))
            if value > self.brush_size_slider.maximum():
                self.brush_size_slider.setMaximum(int(value))
            self.brush_size_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.brush_size_slider_label, self.brush_size_slider)]
