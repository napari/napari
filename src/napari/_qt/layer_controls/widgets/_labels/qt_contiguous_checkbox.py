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
from napari.layers import Labels
from napari.utils.translations import trans


class QtContiguousCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contiguous
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    contiguous_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is contiguous.
    contiguous_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the contiguous model chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.contiguous.connect(self._on_contiguous_change)

        # Setup widgets
        contig_cb = QCheckBox()
        contig_cb.setToolTip(trans._('Contiguous editing'))
        contig_cb.stateChanged.connect(self.change_contig)
        self.contiguous_checkbox = contig_cb
        self._on_contiguous_change()

        self.contiguous_checkbox_label = QtWrappedLabel(trans._('contiguous:'))

    def change_contig(self, state: int) -> None:
        """Toggle contiguous state of label layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of contiguous_checkbox
        """
        self._layer.contiguous = Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_contiguous_change(self) -> None:
        """Receive layer model contiguous change event and update the checkbox."""
        with qt_signals_blocked(self.contiguous_checkbox):
            self.contiguous_checkbox.setChecked(self._layer.contiguous)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.contiguous_checkbox_label, self.contiguous_checkbox)]
