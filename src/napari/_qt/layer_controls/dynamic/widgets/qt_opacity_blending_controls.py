from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from superqt import QEnumComboBox, QLabeledDoubleSlider

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.layers.base._base_constants import Blending
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import (
        QWidget,
    )

    from napari.layers.base.base import Layer


# opaque, minimum, and multiplicative blending do not support changing alpha (opacity)
NO_OPACITY_BLENDING_MODES = {
    str(Blending.MINIMUM),
    str(Blending.OPAQUE),
    str(Blending.MULTIPLICATIVE),
}


class QtOpacityBlendingControls(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between opacity/blending
    layer attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list of napari.layers.Layer
        A list of napari layers.

    Attributes
    ----------
    blend_combobox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blend_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    opacity_slider : superqt.QLabeledDoubleSlider
        Slider controlling opacity of the layer.
    opacity_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    """

    def __init__(self, parent: QWidget, layers: list[Layer]) -> None:
        super().__init__(parent, layers)
        # Setup layer
        for layer in self._layers:
            layer.events.blending.connect(self._on_blending_change)

        # Setup widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(1)
        sld.setSingleStep(0.01)
        sld.setValue(self._layers[0].opacity)
        self.opacity_slider = sld
        for layer in self._layers:
            connect_setattr(self.opacity_slider.valueChanged, layer, 'opacity')
        self._callbacks.append(
            attr_to_settr(
                self._layers[0], 'opacity', self.opacity_slider, 'setValue'
            )
        )
        self.opacity_label = QtWrappedLabel('opacity:')

        blend_combobox = QEnumComboBox(parent, Blending)

        blend_combobox.setCurrentEnum(Blending(self._layers[0].blending))

        blend_combobox.currentEnumChanged.connect(self.change_blending)
        self.blend_combobox = blend_combobox
        self.blend_label = QtWrappedLabel('blending:')

        # opaque and minimum blending do not support changing alpha
        self.opacity_slider.setEnabled(
            all(
                layer.blending not in NO_OPACITY_BLENDING_MODES
                for layer in self._layers
            )
        )
        self.opacity_label.setEnabled(
            all(
                layer.blending not in NO_OPACITY_BLENDING_MODES
                for layer in self._layers
            )
        )

    def change_blending(self, text: str) -> None:
        """Change blending mode on the layer model.

        Parameters
        ----------
        text : str
            Name of blending mode, eg: 'translucent', 'additive', 'opaque'.
        """
        for layer in self._layers:
            layer.blending = self.blend_combobox.currentEnum()
            # opaque and minimum blending do not support changing alpha
        self.opacity_slider.setEnabled(
            all(
                layer.blending not in NO_OPACITY_BLENDING_MODES
                for layer in self._layers
            )
        )
        self.opacity_label.setEnabled(
            all(
                layer.blending not in NO_OPACITY_BLENDING_MODES
                for layer in self._layers
            )
        )

        blending_tooltip = ''
        if any(layer.blending == Blending.MINIMUM for layer in self._layers):
            blending_tooltip = '`minimum` blending mode works best with inverted colormaps with a white background.'
        self.blend_combobox.setToolTip(blending_tooltip)
        self._layers[0].help = blending_tooltip

    def _on_blending_change(self) -> None:
        """Receive layer model blending mode change event and update slider."""
        with ExitStack() as stack:
            for layer in self._layers:
                stack.enter_context(layer.events.blending.blocker())
            self.blend_combobox.setCurrentEnum(
                Blending(self._layers[0].blending)
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.opacity_label, self.opacity_slider),
            (self.blend_label, self.blend_combobox),
        ]
