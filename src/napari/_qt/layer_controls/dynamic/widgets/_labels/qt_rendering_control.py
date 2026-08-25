from __future__ import annotations

from typing import TYPE_CHECKING

from superqt import QEnumComboBox

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelsRendering,
)
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Labels


class QtLabelRenderingControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render the labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Labels]
        A list of napari Labels layers.

    Attributes
    ----------
    iso_gradient_combobox : superqt.QEnumComboBox
        Combobox to control gradient method when isosurface rendering is selected.
    iso_gradient_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the chooser widget of the gradient to use when labels are using isosurface rendering.
    rendering_combobox : superqt.QEnumComboBox
        Combobox to control current label render method.
    rendering_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way labels should be rendered chooser widget.
    """

    _layers: list[Labels]

    def __init__(
        self, layers: list[Labels], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.rendering.connect(self._on_rendering_change)
            layer.events.iso_gradient_mode.connect(
                self._on_iso_gradient_mode_change
            )

        # Setup widgets
        rendering_combobox = QEnumComboBox(enum_class=LabelsRendering)
        rendering_combobox.setCurrentEnum(
            LabelsRendering(self._layers[0].rendering)
        )
        self.rendering_combobox = rendering_combobox
        for layer in self._layers:
            connect_setattr(
                rendering_combobox.currentEnumChanged, layer, 'rendering'
            )
        self.rendering_combobox_label = QtWrappedLabel('rendering:')

        iso_gradient_combobox = QEnumComboBox(
            enum_class=IsoCategoricalGradientMode
        )
        iso_gradient_combobox.setCurrentEnum(
            IsoCategoricalGradientMode(self._layers[0].iso_gradient_mode)
        )
        for layer in self._layers:
            connect_setattr(
                iso_gradient_combobox.currentEnumChanged,
                layer,
                'iso_gradient_mode',
            )
            iso_gradient_combobox.setEnabled(
                layer.rendering == LabelsRendering.ISO_CATEGORICAL
            )
        self.iso_gradient_combobox = iso_gradient_combobox
        self.iso_gradient_combobox_label = QtWrappedLabel('gradient\nmode:')

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        rendering_mode = LabelsRendering(self._layers[0].rendering)

        with qt_signals_blocked(self.rendering_combobox):
            self.rendering_combobox.setCurrentEnum(rendering_mode)

        with qt_signals_blocked(self.iso_gradient_combobox):
            self.iso_gradient_combobox.setEnabled(
                rendering_mode == LabelsRendering.ISO_CATEGORICAL
            )

    def _on_iso_gradient_mode_change(self):
        """Receive layer model iso_gradient_mode change event and update dropdown menu."""
        with qt_signals_blocked(self.iso_gradient_combobox):
            self.iso_gradient_combobox.setCurrentEnum(
                IsoCategoricalGradientMode(self._layers[0].iso_gradient_mode)
            )

    def _change_ndisplay(self, ndisplay: int) -> None:
        if ndisplay == 3:
            self.rendering_combobox.show()
            self.rendering_combobox_label.show()
            self.iso_gradient_combobox.show()
            self.iso_gradient_combobox_label.show()
        else:
            self.rendering_combobox.hide()
            self.rendering_combobox_label.hide()
            self.iso_gradient_combobox.hide()
            self.iso_gradient_combobox_label.hide()

    def get_widget_controls(self) -> list[tuple[QWidget, ...]]:
        return [
            (self.rendering_combobox_label, self.rendering_combobox),
            (self.iso_gradient_combobox_label, self.iso_gradient_combobox),
        ]
