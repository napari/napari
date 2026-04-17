from qtpy.QtWidgets import QWidget
from superqt import QEnumComboBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Labels
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelsRendering,
)
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtLabelRenderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render the labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels Labels layer.

    Attributes
    ----------
    iso_gradient_combobox : superqt.QEnumComboBox
        Combobox to control gradient method when isosurface rendering is selected.
    iso_gradient_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the chooser widget of the gradient to use when labels are using isosurface rendering.
    render_combobox : superqt.QEnumComboBox
        Combobox to control current label render method.
    render_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the way labels should be rendered chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.rendering.connect(self._on_rendering_change)
        self._layer.events.iso_gradient_mode.connect(
            self._on_iso_gradient_mode_change
        )

        # Setup widgets
        render_combobox = QEnumComboBox(enum_class=LabelsRendering)
        render_combobox.setCurrentEnum(LabelsRendering(self._layer.rendering))
        self.render_combobox = render_combobox
        connect_setattr(
            render_combobox.currentEnumChanged, self._layer, 'rendering'
        )
        self.render_combobox_label = QtWrappedLabel(trans._('rendering:'))

        iso_gradient_combobox = QEnumComboBox(
            enum_class=IsoCategoricalGradientMode
        )
        iso_gradient_combobox.setCurrentEnum(
            IsoCategoricalGradientMode(self._layer.iso_gradient_mode)
        )
        connect_setattr(
            iso_gradient_combobox.currentEnumChanged,
            self._layer,
            'iso_gradient_mode',
        )
        iso_gradient_combobox.setEnabled(
            self._layer.rendering == LabelsRendering.ISO_CATEGORICAL
        )
        self.iso_gradient_combobox = iso_gradient_combobox
        self.iso_gradient_combobox_label = QtWrappedLabel(
            trans._('gradient\nmode:')
        )

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        rendering_mode = LabelsRendering(self._layer.rendering)

        with qt_signals_blocked(self.render_combobox):
            self.render_combobox.setCurrentEnum(rendering_mode)

        with qt_signals_blocked(self.iso_gradient_combobox):
            self.iso_gradient_combobox.setEnabled(
                rendering_mode == LabelsRendering.ISO_CATEGORICAL
            )

    def _on_iso_gradient_mode_change(self):
        """Receive layer model iso_gradient_mode change event and update dropdown menu."""
        with qt_signals_blocked(self.iso_gradient_combobox):
            self.iso_gradient_combobox.setCurrentEnum(
                IsoCategoricalGradientMode(self._layer.iso_gradient_mode)
            )

    def _on_display_change_hide(self):
        self.render_combobox.hide()
        self.render_combobox_label.hide()
        self.iso_gradient_combobox.hide()
        self.iso_gradient_combobox_label.hide()

    def _on_display_change_show(self):
        self.render_combobox.show()
        self.render_combobox_label.show()
        self.iso_gradient_combobox.show()
        self.iso_gradient_combobox_label.show()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.render_combobox_label, self.render_combobox),
            (self.iso_gradient_combobox_label, self.iso_gradient_combobox),
        ]
