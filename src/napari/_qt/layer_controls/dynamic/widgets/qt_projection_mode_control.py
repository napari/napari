from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers.base.base import Image, Points, Vectors


class QtProjectionModeControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer projection
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Image | napari.layers.Points | napari.layers.Vectors]
        A list of Image, Points and Vectors napari layers.

    Attributes
    ----------
    projection_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current projection mode of the layer.
    projection_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the projection mode chooser widget.
    """

    _layers: list[Image | Points | Vectors]

    def __init__(
        self,
        layers: list[Image | Points | Vectors],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.projection_mode.connect(
                self._on_projection_mode_change
            )

        # Setup widgets
        proj_modes = [
            i.lower()
            for i in set.intersection(
                *(
                    set(layer._projectionclass.__members__)
                    for layer in self._layers
                )
            )
        ]
        self.projection_combobox = QComboBox(parent)
        self.projection_combobox.addItems(proj_modes)
        for layer in self._layers:
            connect_setattr(
                self.projection_combobox.currentTextChanged,
                layer,
                'projection_mode',
            )

        self._on_projection_mode_change()

        self.projection_combobox_label = QtWrappedLabel('projection mode:')

    def _on_projection_mode_change(self) -> None:
        with qt_signals_blocked(self.projection_combobox):
            self.projection_combobox.setCurrentText(
                str(self._layers[0].projection_mode)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.projection_combobox_label, self.projection_combobox)]
