from collections.abc import Iterable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from superqt import QLabeledSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Shapes
from napari.utils.events.event_utils import connect_setattr


class QtEdgeWidthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    width layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.

    Attributes
    ----------
    border_width_slider : superqt.QLabeledDoubleSlider
        Slider controlling line edge width of layer.
    border_width_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge width widget.
    """

    def __init__(self, parent: QWidget, layer: Shapes) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.border_width.connect(self._on_border_width_change)

        # Setup widgets
        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self._layer.current_border_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        self.border_width_slider = sld
        connect_setattr(
            self.border_width_slider.valueChanged,
            self._layer,
            'current_border_width',
            convert_fun=float,
        )
        self.border_width_slider.setToolTip(
            'Set the edge width of currently selected shapes and any added afterwards.'
        )
        self.border_width_label = QtWrappedLabel('edge width:')

    def _on_border_width_change(self) -> None:
        """Receive layer model edge line width change event and update slider."""
        with qt_signals_blocked(self.border_width_slider):
            value = self._layer.current_border_width
            value = np.clip(int(value), 0, 40)
            self.border_width_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.border_width_label, self.border_width_slider)]
