from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtPreserveLabelsCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute to
    preserve existing labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    preserve_labels_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if existing labels are preserved.
    preserve_labels_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should preserve labels chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.preserve_labels.connect(
            self._on_preserve_labels_change
        )

        # Setup widgets
        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            trans._('Preserve existing labels while painting')
        )
        preserve_labels_cb.stateChanged.connect(self.change_preserve_labels)
        self.preserve_labels_checkbox = preserve_labels_cb
        self._on_preserve_labels_change()

        self.preserve_labels_checkbox_label = QtWrappedLabel(
            trans._('preserve\nlabels:')
        )

    def change_preserve_labels(self, state) -> None:
        """Toggle preserve_labels state of label layer.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of preserve_labels_checkbox
        """
        self._layer.preserve_labels = (
            Qt.CheckState(state) == Qt.CheckState.Checked
        )

    def _on_preserve_labels_change(self) -> None:
        """Receive layer model preserve_labels event and update the checkbox."""
        with self._layer.events.preserve_labels.blocker():
            self.preserve_labels_checkbox.setChecked(
                self._layer.preserve_labels
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.preserve_labels_checkbox_label,
                self.preserve_labels_checkbox,
            )
        ]
