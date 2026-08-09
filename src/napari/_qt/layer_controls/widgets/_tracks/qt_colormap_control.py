from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.utils.colormaps import AVAILABLE_COLORMAPS

if TYPE_CHECKING:
    from napari.layers import Tracks


class QtColormapComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer colormaps
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    colormap_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current colormap of the layer.
    colormap_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the colormap chooser widget.
    """

    def __init__(self, parent: QWidget, layers: list[Tracks]) -> None:
        super().__init__(parent, layers)
        self._layers = layers
        # Setup layer
        for layer in self._layers:
            layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        self.colormap_combobox = QComboBox()
        for name, colormap in AVAILABLE_COLORMAPS.items():
            display_name = colormap._display_name
            self.colormap_combobox.addItem(display_name, name)
        self.colormap_combobox.currentTextChanged.connect(self.change_colormap)

        self.colormap_combobox_label = QtWrappedLabel('colormap:')

        self._on_colormap_change()

    def change_colormap(self, colormap: str):
        for layer in self._layers:
            layer.colormap = self.colormap_combobox.currentData()

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update combobox."""
        with qt_signals_blocked(self.colormap_combobox):
            self.colormap_combobox.setCurrentIndex(
                self.colormap_combobox.findData(self._layers[0].colormap)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.colormap_combobox_label, self.colormap_combobox)]
