from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtOutSliceCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer out of slice
    display attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    out_of_slice_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    out_of_slice_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the out of slice display enablement chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.out_of_slice_display.connect(
            self._on_out_of_slice_display_change
        )

        # Setup widgets
        self.out_of_slice_checkbox = QCheckBox()
        self.out_of_slice_checkbox.setToolTip(trans._('Out of slice display'))
        self.out_of_slice_checkbox.setChecked(self._layer.out_of_slice_display)
        self.out_of_slice_checkbox.stateChanged.connect(
            self.change_out_of_slice
        )

        self.out_of_slice_checkbox_label = QtWrappedLabel(
            trans._('out of slice:')
        )

    def change_out_of_slice(self, state) -> None:
        """Toggleout of slice display of points layer.

        Parameters
        ----------
        state : Qt.CheckState
            Checkbox indicating whether to render out of slice.
        """
        # needs cast to bool for Qt6
        with self._layer.events.out_of_slice_display.blocker(
            self._on_out_of_slice_display_change
        ):
            self._layer.out_of_slice_display = (
                Qt.CheckState(state) == Qt.CheckState.Checked
            )

    def _on_out_of_slice_display_change(self) -> None:
        """Receive layer model out_of_slice_display change event and update checkbox."""
        with qt_signals_blocked(self.out_of_slice_checkbox):
            self.out_of_slice_checkbox.setChecked(
                self._layer.out_of_slice_display
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.out_of_slice_checkbox_label, self.out_of_slice_checkbox)]
