from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
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


class QtTailLengthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current tail length
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari layer.

    Attributes
    ----------
    tail_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail length of the layer.
    tail_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail length chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.tail_length.connect(self._on_tail_length_change)

        # Setup widgets
        # slider for track tail length
        self.tail_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_length_slider.setMinimum(1)
        self.tail_length_slider.setMaximum(self._layer._max_length)
        self.tail_length_slider.setSingleStep(1)
        connect_setattr(
            self.tail_length_slider.valueChanged, self._layer, 'tail_length'
        )

        self.tail_length_slider_label = QtWrappedLabel(trans._('tail length:'))

        self._on_tail_length_change()

    def _on_tail_length_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.tail_length_slider):
            value = self._layer.tail_length
            if value > self.tail_length_slider.maximum():
                self.tail_length_slider.setMaximum(self._layer._max_length)
            self.tail_length_slider.setValue(value)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.tail_length_slider_label, self.tail_length_slider)]


class QtTailWidthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current tail width
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    tail_width_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail width of the layer.
    tail_width_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail width chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.tail_width.connect(self._on_tail_width_change)
        self._layer.events.data.connect(self._on_data_change)

        # Setup widgets
        # slider for track edge width
        self.tail_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_width_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_width_slider.setMinimum(1)
        self.tail_width_slider.setMaximum(int(self._layer._max_width))
        self.tail_width_slider.setSingleStep(1)
        connect_setattr(
            self.tail_width_slider.valueChanged, self._layer, 'tail_width'
        )

        self.tail_width_slider_label = QtWrappedLabel(trans._('tail width:'))

        self._on_tail_width_change()

    def _on_tail_width_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.tail_width_slider):
            value = int(self._layer.tail_width)
            self.tail_width_slider.setValue(value)

    def _on_data_change(self) -> None:
        self.tail_width_slider.setMaximum(int(self._layer._max_width))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.tail_width_slider_label, self.tail_width_slider)]


class QtTailDisplayCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the tail should be
    displayed attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    tail_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if tails of the layer should be shown.
    tail_width_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the tails chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        # NOTE(arl): there are no events fired for changing checkbox (layer `display_tail` attribute)

        # Setup widgets
        self.tail_checkbox = QCheckBox()
        self.tail_checkbox.setChecked(True)
        self.tail_checkbox.stateChanged.connect(self.change_display_tail)

        self.tail_checkbox_label = QtWrappedLabel(trans._('tail:'))

    def change_display_tail(self, state: Qt.CheckState) -> None:
        self._layer.display_tail = self.tail_checkbox.isChecked()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.tail_checkbox_label, self.tail_checkbox)]
