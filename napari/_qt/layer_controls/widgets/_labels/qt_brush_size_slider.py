from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
)
from superqt import QLabeledSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtBrushSizeSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current brush
    size attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    brush_size_slider : superqt.QLabeledDoubleSlider
        Slider controlling current brush size of the layer.
    brush_size_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the brush size chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.brush_size.connect(self._on_brush_size_change)

        # Setup widgets
        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        sld.valueChanged.connect(self.change_size)
        self.brush_size_slider = sld
        self._on_brush_size_change()

        self.brush_size_slider_label = QtWrappedLabel(trans._('brush size:'))

    def change_size(self, value: float) -> None:
        """Change paint brush size.

        Parameters
        ----------
        value : float
            Size of the paint brush.
        """
        self._layer.brush_size = value

    def _on_brush_size_change(self) -> None:
        """Receive layer model brush size change event and update the slider."""
        with self._layer.events.brush_size.blocker():
            value = self._layer.brush_size
            value = np.maximum(1, int(value))
            if value > self.brush_size_slider.maximum():
                self.brush_size_slider.setMaximum(int(value))
            self.brush_size_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.brush_size_slider_label, self.brush_size_slider)]
