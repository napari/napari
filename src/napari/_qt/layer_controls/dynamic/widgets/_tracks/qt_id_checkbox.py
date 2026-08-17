from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Tracks


class QtIdCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the id should be
    displayed attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Tracks]
        A list of napari Tracks layers.

    Attributes
    ----------
    id_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if id of the layer should be shown.
    id_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the id chooser widget.
    """

    _layers: list[Tracks]

    def __init__(
        self, layers: list[Tracks], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        # NOTE(arl): there are no events fired for changing checkbox (layer `display_id` attribute)

        # Setup widgets
        self.id_checkbox = QCheckBox()
        for layer in self._layers:
            connect_setattr(
                self.id_checkbox.stateChanged,
                layer,
                'display_id',
                convert_fun=checked_to_bool,
            )

        self.id_checkbox_label = QtWrappedLabel('show ID:')

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.id_checkbox_label, self.id_checkbox)]
