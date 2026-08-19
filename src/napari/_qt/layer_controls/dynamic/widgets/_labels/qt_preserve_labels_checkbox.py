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
from napari._qt.utils import attr_to_settr, checked_to_bool
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Labels


class QtPreserveLabelsCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute to
    preserve existing labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Labels]
        A list of napari Labels layers.

    Attributes
    ----------
    preserve_labels_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if existing labels are preserved.
    preserve_labels_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should preserve labels chooser widget.
    """

    _layers: list[Labels]

    def __init__(
        self, layers: list[Labels], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup widgets
        preserve_labels_cb = QCheckBox()
        preserve_labels_cb.setToolTip(
            'Preserve existing labels while painting'
        )
        preserve_labels_cb.setChecked(self._layers[0].preserve_labels)
        self._callbacks.append(
            attr_to_settr(
                self._layers[0],
                'preserve_labels',
                preserve_labels_cb,
                'setChecked',
            )
        )
        for layer in self._layers:
            connect_setattr(
                preserve_labels_cb.stateChanged,
                layer,
                'preserve_labels',
                convert_fun=checked_to_bool,
            )
        self.preserve_labels_checkbox = preserve_labels_cb

        self.preserve_labels_checkbox_label = QtWrappedLabel(
            'preserve\nlabels:'
        )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.preserve_labels_checkbox_label,
                self.preserve_labels_checkbox,
            )
        ]
