from qtpy.QtWidgets import (
    QWidget,
)
from superqt import QEnumComboBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.layers import Labels
from napari.layers.labels._labels_constants import (
    LabelColorMode,
)
from napari.utils import CyclicLabelColormap


class QtColorModeComboBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer color
    mode attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    color_mode_combobox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current color mode of the layer.
    color_mode_combobox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color mode chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.colormap.connect(self._on_colormap_change)

        # Setup widgets
        color_mode_comboBox = QEnumComboBox(enum_class=LabelColorMode)
        self.color_mode_combobox = color_mode_comboBox
        self._on_colormap_change()
        color_mode_comboBox.activated.connect(self.change_color_mode)

        self.color_mode_combobox_label = QtWrappedLabel('color mode:')

    def change_color_mode(self) -> None:
        """Change color mode of label layer"""
        if self.color_mode_combobox.currentEnum() == LabelColorMode.AUTO.value:
            self._layer.colormap = self._layer._original_random_colormap
        else:
            self._layer.colormap = self._layer._direct_colormap

    def _on_colormap_change(self) -> None:
        enable_combobox = not self._layer._is_default_colors(
            self._layer._direct_colormap.color_dict
        )
        self.color_mode_combobox.setEnabled(enable_combobox)
        if not enable_combobox:
            self.color_mode_combobox.setToolTip(
                'Layer needs a user-set DirectLabelColormap to enable direct '
                'mode.'
            )
        if isinstance(self._layer.colormap, CyclicLabelColormap):
            self.color_mode_combobox.setCurrentEnum(LabelColorMode.AUTO)
        else:
            self.color_mode_combobox.setCurrentEnum(LabelColorMode.DIRECT)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.color_mode_combobox_label, self.color_mode_combobox)]
