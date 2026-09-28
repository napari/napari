from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QSlider,
    QWidget,
)

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Tracks


class QtTailLengthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current tail length
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Tracks]
        A list of napari Tracks layers.

    Attributes
    ----------
    tail_length_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail length of the layer.
    tail_length_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail length chooser widget.
    """

    _layers: list[Tracks]

    def __init__(
        self, layers: list[Tracks], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.tail_length.connect(self._on_tail_length_change)

        # Setup widgets
        # slider for track tail length
        self.tail_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_length_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_length_slider.setMinimum(1)
        self.tail_length_slider.setMaximum(self._layers[0]._max_length)
        self.tail_length_slider.setSingleStep(1)
        for layer in self._layers:
            connect_setattr(
                self.tail_length_slider.valueChanged, layer, 'tail_length'
            )

        self.tail_length_slider_label = QtWrappedLabel('tail length:')
        self._layers[0].events.tail_length.connect(self._on_tail_length_change)
        self._on_tail_length_change()

    def _on_tail_length_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.tail_length_slider):
            value = int(self._layers[0].tail_length)
            if value > self.tail_length_slider.maximum():
                self.tail_length_slider.setMaximum(
                    int(self._layers[0]._max_length)
                )
            self.tail_length_slider.setValue(value)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.tail_length_slider_label, self.tail_length_slider)]


class QtTailWidthSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current tail width
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Tracks]
        A list of napari Tracks layers.

    Attributes
    ----------
    tail_width_slider : qtpy.QtWidgets.QSlider
        Slider controlling tail width of the layer.
    tail_width_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the tail width chooser widget.
    """

    _layers: list[Tracks]

    def __init__(
        self, layers: list[Tracks], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.tail_width.connect(self._on_tail_width_change)

        # Setup widgets
        # slider for track edge width
        self.tail_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.tail_width_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tail_width_slider.setMinimum(1)
        self.tail_width_slider.setMaximum(int(self._layers[0]._max_width))
        self.tail_width_slider.setSingleStep(1)
        for layer in self._layers:
            connect_setattr(
                self.tail_width_slider.valueChanged, layer, 'tail_width'
            )

        self.tail_width_slider_label = QtWrappedLabel('tail width:')

        self._on_tail_width_change()

    def _on_tail_width_change(self) -> None:
        """Receive layer model track line width change event and update slider."""
        with qt_signals_blocked(self.tail_width_slider):
            value = int(self._layers[0].tail_width)
            if value > self.tail_width_slider.maximum():
                self.tail_width_slider.setMaximum(
                    int(self._layers[0]._max_width)
                )
            self.tail_width_slider.setValue(value)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.tail_width_slider_label, self.tail_width_slider)]


class QtTailDisplayCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the tail should be
    displayed attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Tracks]
        A list of napari Tracks layers.

    Attributes
    ----------
    tail_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if tails of the layer should be shown.
    tail_width_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the tails chooser widget.
    """

    _layers: list[Tracks]

    def __init__(
        self, layers: list[Tracks], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        # NOTE(arl): there are no events fired for changing checkbox (layer `display_tail` attribute)

        # Setup widgets
        self.tail_checkbox = QCheckBox()
        self.tail_checkbox.setChecked(True)
        for layer in self._layers:
            connect_setattr(
                self.tail_checkbox.stateChanged,
                layer,
                'display_tail',
                convert_fun=checked_to_bool,
            )

        self._layers[0].events.display_tail.connect(self._set_display_tail)
        self.tail_checkbox_label = QtWrappedLabel('tail:')

    def _set_display_tail(self) -> None:
        """Receive layer model track line width change event and update checkbox."""
        self.tail_checkbox.setChecked(self._layers[0].display_tail)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.tail_checkbox_label, self.tail_checkbox)]
