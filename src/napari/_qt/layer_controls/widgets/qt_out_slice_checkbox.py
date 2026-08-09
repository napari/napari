from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Points, Vectors


class QtOutSliceCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer out of slice
    display attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Points | napari.layers.Vectors]
        A list of Points and Vectors napari layers.

    Attributes
    ----------
    out_of_slice_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    out_of_slice_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the out of slice display enablement chooser widget.
    """

    def __init__(
        self, parent: QWidget, layers: list[Points | Vectors]
    ) -> None:
        super().__init__(parent, layers)
        # Setup layer
        self._layers = layers

        for layer in self._layers:
            layer.events.out_of_slice_display.connect(
                self._on_out_of_slice_display_change
            )

        # Setup widgets
        self.out_of_slice_checkbox = QCheckBox()
        self.out_of_slice_checkbox.setToolTip('Out of slice display')
        self.out_of_slice_checkbox.setChecked(
            self._layers[0].out_of_slice_display
        )
        for layer in self._layers:
            connect_setattr(
                self.out_of_slice_checkbox.stateChanged,
                layer,
                'out_of_slice_display',
                convert_fun=checked_to_bool,
            )

        self.out_of_slice_checkbox_label = QtWrappedLabel('out of slice:')

    def _on_out_of_slice_display_change(self) -> None:
        """Receive layer model out_of_slice_display change event and update checkbox."""
        with qt_signals_blocked(self.out_of_slice_checkbox):
            self.out_of_slice_checkbox.setChecked(
                self._layers[0].out_of_slice_display
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.out_of_slice_checkbox_label, self.out_of_slice_checkbox)]
