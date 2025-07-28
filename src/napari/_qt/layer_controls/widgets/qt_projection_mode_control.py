from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Image, Points, Vectors
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtProjectionModeControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer projection
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Image | napari.layers.Points | napari.layers.Vectors
        An instance of an Image, Points or Vectors napari layer.

    Attributes
    ----------
    projection_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current projection mode of the layer.
    projection_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the projection mode chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Image | Points | Vectors
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.projection_mode.connect(
            self._on_projection_mode_change
        )

        # Setup widgets
        proj_modes = [i.value for i in self._layer._projectionclass]
        self.projection_combobox = QComboBox(parent)
        self.projection_combobox.addItems(proj_modes)
        connect_setattr(
            self.projection_combobox.currentTextChanged,
            self._layer,
            'projection_mode',
        )

        self._on_projection_mode_change()

        self.projection_combobox_label = QtWrappedLabel(
            trans._('projection mode:')
        )

    def _on_projection_mode_change(self) -> None:
        with qt_signals_blocked(self.projection_combobox):
            self.projection_combobox.setCurrentText(
                str(self._layer.projection_mode)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.projection_combobox_label, self.projection_combobox)]
