from __future__ import annotations

from typing import TYPE_CHECKING

from superqt import QEnumComboBox

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.surface._surface_constants import Shading

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Surface


class QtShadingComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer shading
    value attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Surface]
        A list of napari Surface layers.

    Attributes
    ----------
    shading_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current shading value of the layer.
    shading_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the shading value chooser widget.
    """

    _layers: list[Surface]

    def __init__(
        self, layers: list[Surface], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        self._layers = layers
        # Setup layer
        for layer in self._layers:
            layer.events.shading.connect(self._on_shading_change)

        # Setup widgets
        shading_comboBox = QEnumComboBox(parent, Shading)
        shading_comboBox.setCurrentEnum(Shading(self._layers[0].shading))
        shading_comboBox.currentEnumChanged.connect(self.change_shading)
        self.shading_combobox = shading_comboBox

        self.shading_combobox_label = QtWrappedLabel('shading:')

    def change_shading(self, text: str) -> None:
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        for layer in self._layers:
            with layer.events.shading.blocker(self._on_shading_change):
                layer.shading = self.shading_combobox.currentEnum()

    def _on_shading_change(self) -> None:
        """Receive layer model shading change event and update combobox."""
        with qt_signals_blocked(self.shading_combobox):
            self.shading_combobox.setCurrentEnum(
                Shading(self._layers[0].shading)
            )

    def get_widget_controls(self) -> list[tuple[QWidget, ...]]:
        return [(self.shading_combobox_label, self.shading_combobox)]
