from __future__ import annotations

from typing import TYPE_CHECKING

from superqt import QEnumComboBox

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers.labels._labels_constants import (
    LabelColorMode,
)
from napari.utils import CyclicLabelColormap

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Labels


class QtColorModeComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer color
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Labels]
        A list of napari Labels layers.

    Attributes
    ----------
    color_mode_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current color mode of the layer.
    color_mode_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color mode chooser widget.
    """

    _layers: list[Labels]

    def __init__(
        self, layers: list[Labels], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        color_mode_comboBox = QEnumComboBox(enum_class=LabelColorMode)
        self.color_mode_combobox = color_mode_comboBox
        self._on_colormap_change()
        color_mode_comboBox.activated.connect(self.change_color_mode)

        self.color_mode_combobox_label = QtWrappedLabel('color mode:')

    def change_color_mode(self) -> None:
        """Change color mode of label layer"""
        for layer in self._layers:
            if (
                self.color_mode_combobox.currentEnum()
                == LabelColorMode.AUTO.value
            ):
                layer.colormap = layer._original_random_colormap
            else:
                layer.colormap = layer._direct_colormap

    def _on_colormap_change(self) -> None:
        enable_combobox = not self._layers[0]._is_default_colors(
            self._layers[0]._direct_colormap.color_dict
        )
        self.color_mode_combobox.setEnabled(enable_combobox)
        if not enable_combobox:
            self.color_mode_combobox.setToolTip(
                'Layer needs a user-set DirectLabelColormap to enable direct '
                'mode.'
            )
        if isinstance(self._layers[0].colormap, CyclicLabelColormap):
            self.color_mode_combobox.setCurrentEnum(LabelColorMode.AUTO)
        else:
            self.color_mode_combobox.setCurrentEnum(LabelColorMode.DIRECT)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.color_mode_combobox_label, self.color_mode_combobox)]
