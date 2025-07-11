from typing import Optional

from qtpy.QtWidgets import QWidget
from superqt import QEnumComboBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.base.base import Layer
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelsRendering,
)
from napari.utils.translations import trans


class QtLabelRenderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute for
    the method to render the labels and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

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

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.rendering.connect(self._on_rendering_change)
        self._layer.events.iso_gradient_mode.connect(
            self._on_iso_gradient_mode_change
        )

        # Setup widgets
        render_combobox = QEnumComboBox(enum_class=LabelsRendering)
        render_combobox.setCurrentEnum(LabelsRendering(self._layer.rendering))
        render_combobox.currentEnumChanged.connect(self.change_rendering)
        self.render_combobox = render_combobox
        self.render_combobox_label = QtWrappedLabel(trans._('rendering:'))

        iso_gradient_combobox = QEnumComboBox(
            enum_class=IsoCategoricalGradientMode
        )
        iso_gradient_combobox.setCurrentEnum(
            IsoCategoricalGradientMode(self._layer.iso_gradient_mode)
        )
        iso_gradient_combobox.currentEnumChanged.connect(
            self.change_iso_gradient_mode
        )
        iso_gradient_combobox.setEnabled(
            self._layer.rendering == LabelsRendering.ISO_CATEGORICAL
        )
        self.iso_gradient_combobox = iso_gradient_combobox
        self.iso_gradient_combobox_label = QtWrappedLabel(
            trans._('gradient\nmode:')
        )

    def change_rendering(self, rendering_mode: LabelsRendering):
        """Change rendering mode for image display.

        Parameters
        ----------
        rendering_mode : LabelsRendering
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * iso_categorical: isosurface for categorical data (e.g., labels).
              Cast a ray until a value greater than zero is encountered. At that
              location, lighning calculations are performed to give the visual
              appearance of a surface.
        """
        self.iso_gradient_combobox.setEnabled(
            rendering_mode == LabelsRendering.ISO_CATEGORICAL
        )
        self._layer.rendering = rendering_mode

    def change_iso_gradient_mode(
        self, gradient_mode: IsoCategoricalGradientMode
    ):
        """Change gradient mode for isosurface rendering.

        Parameters
        ----------
        gradient_mode : IsoCategoricalGradientMode
            Gradient mode for the isosurface rendering method.
            Selects the finite-difference gradient method for the isosurface shader:
            * fast: simple finite difference gradient along each axis
            * smooth: isotropic Sobel gradient, smoother but more computationally expensive
        """
        self._layer.iso_gradient_mode = gradient_mode

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self._layer.events.rendering.blocker():
            self.render_combobox.setCurrentEnum(
                LabelsRendering(self._layer.rendering)
            )

    def _on_iso_gradient_mode_change(self):
        """Receive layer model iso_gradient_mode change event and update dropdown menu."""
        with self._layer.events.iso_gradient_mode.blocker():
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
