from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr, checked_to_bool
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Labels


class QtDisplaySelectedLabelCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute to
    only display selected label and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    selected_color_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if only currently selected label is shown.
    selected_color_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should show only currently selected label chooser widget.
    """

    def __init__(self, parent: QWidget, layers: list[Labels]) -> None:
        super().__init__(parent, layers)
        self._layers = layers
        # Setup widgets
        selected_color_checkbox = QCheckBox()
        selected_color_checkbox.setToolTip('Display only selected label')
        selected_color_checkbox.setChecked(self._layers[0].show_selected_label)
        for layer in self._layers:
            self._callbacks.append(
                attr_to_settr(
                    layer,
                    'show_selected_label',
                    selected_color_checkbox,
                    'setChecked',
                )
            )
            connect_setattr(
                selected_color_checkbox.stateChanged,
                layer,
                'show_selected_label',
                convert_fun=checked_to_bool,
            )
        self.selected_color_checkbox = selected_color_checkbox

        self.selected_color_checkbox_label = QtWrappedLabel('show\nselected:')

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.selected_color_checkbox_label, self.selected_color_checkbox)
        ]
