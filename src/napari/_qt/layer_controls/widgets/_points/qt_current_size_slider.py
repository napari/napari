import contextlib

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
from napari._qt.utils import qt_signals_blocked
from napari.layers import Points
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtCurrentSizeSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current
    size layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    size_slider : superqt.QLabeledDoubleSlider
        Slider controlling current size attribute of the layer.
    size_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the size chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Points) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.size.connect(self._on_current_size_change)
        self._layer.events.current_size.connect(self._on_current_size_change)

        # Setup widgets
        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setToolTip(
            trans._(
                'Change the size of currently selected points and any added afterwards.'
            )
        )
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        if self._layer.size.size:
            max_value = max(100, int(np.max(self._layer.size)) + 1)
        else:
            max_value = 100
        sld.setMaximum(max_value)
        sld.setSingleStep(1)
        value = self._layer.current_size
        sld.setValue(int(value))
        self.size_slider = sld
        connect_setattr(
            self.size_slider.valueChanged, self._layer, 'current_size'
        )

        self.size_slider_label = QtWrappedLabel(trans._('point size:'))

    def _on_current_size_change(self) -> None:
        """Receive layer model size change event and update point size slider."""
        with qt_signals_blocked(self.size_slider):
            value = self._layer.current_size
            min_val = min(value) if isinstance(value, list) else value
            max_val = max(value) if isinstance(value, list) else value
            if min_val < self.size_slider.minimum():
                self.size_slider.setMinimum(max(1, int(min_val - 1)))
            if max_val > self.size_slider.maximum():
                self.size_slider.setMaximum(int(max_val + 1))
            with contextlib.suppress(TypeError):
                self.size_slider.setValue(int(value))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.size_slider_label, self.size_slider)]
