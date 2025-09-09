from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QSlider,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Tracks
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtHeadLengthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current head length
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    head_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling head length of the layer.
    head_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the head length chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.head_length.connect(self._on_head_length_change)

        # Setup widgets
        # slider for track head length
        self.head_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.head_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.head_length_slider.setMinimum(0)
        self.head_length_slider.setMaximum(self._layer._max_length)
        self.head_length_slider.setSingleStep(1)
        connect_setattr(
            self.head_length_slider.valueChanged, self._layer, 'head_length'
        )

        self.head_length_slider_label = QtWrappedLabel(trans._('head length:'))

    def _on_head_length_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.head_length_slider):
            value = self._layer.head_length
            if value > self.head_length_slider.maximum():
                self.head_length_slider.setMaximum(self._layer._max_length)
            self.head_length_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.head_length_slider_label, self.head_length_slider)]
