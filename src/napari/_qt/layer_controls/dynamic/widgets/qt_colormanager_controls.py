from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox
from superqt import QEnumComboBox

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.utils._color_manager_constants import ColorMode
from napari.layers.utils.layer_utils import _unique_element
from napari.utils.color import ColorValue

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Points, Shapes

_NONE_STRING = '<---->'


class QtColorManagerControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between layers'
    colormanager attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Points]
        A list of napari Points layers.
    toolip : str
        String to use for the tooltip of the face color edit widget.

    Attributes
    ----------
    color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current face color of the layer.
    color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current face color chooser widget.
    """

    _layers: list[Points | Shapes]

    def __init__(
        self,
        layers: list[Points | Shapes],
        display_name: str,
        colormanager_attribute: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(layers, parent)
        self._colormanager_attribute = colormanager_attribute

        # create controls
        self.color_mode_label = QtWrappedLabel(f'{display_name} color mode:')
        self.color_mode_combobox = QEnumComboBox(parent)
        self.color_mode_combobox.setEnumClass(ColorMode)

        self.color_label = QtWrappedLabel(f'{display_name} color:')
        self.color_edit = QColorSwatchEdit(
            tooltip=f'Click to set the {display_name} color of the current selection and anything added afterwards.',
        )

        self.feature_label = QtWrappedLabel(f'{display_name} color feature:')
        self.color_feature_combobox = QComboBox(parent)

        # initialize values
        self._on_color_mode_change()
        if self.color_mode_combobox.currentEnum() != ColorMode.direct:
            self._on_color_feature_changed()
        self._on_current_color_change()

        # connect all the events both ways
        self.color_mode_combobox.currentTextChanged.connect(
            self.change_color_mode
        )
        self.color_edit.color_changed.connect(self.change_current_color)
        self.color_feature_combobox.currentTextChanged.connect(
            self.change_color_feature
        )

        for layer in self._layers:
            getattr(layer, colormanager_attribute).events.color_mode.connect(
                self._on_color_mode_change
            )
            getattr(
                layer, colormanager_attribute
            ).events.current_color.connect(self._on_current_color_change)
            layer.events.features.connect(self._on_layer_features_change)

    def change_color_mode(self, mode: ColorMode):
        with qt_signals_blocked(self.color_mode_combobox):
            for layer in self._layers:
                manager = getattr(layer, self._colormanager_attribute)
                # if no feature is already selected, do nothing. Let the
                # user select a feature first!
                if manager.color_properties is not None:
                    manager.color_mode = mode

            self._on_color_feature_changed()
            self._update_visible_widgets(mode)

    def _on_color_mode_change(self):
        with qt_signals_blocked(self.color_mode_combobox):
            mode = _unique_element(
                [
                    getattr(layer, self._colormanager_attribute).color_mode
                    for layer in self._layers
                ]
            )
            index = self.color_mode_combobox.findText(
                mode, Qt.MatchFlag.MatchFixedString
            )
            self.color_mode_combobox.setCurrentIndex(index)

            self._update_visible_widgets(mode)

    def change_current_color(self, color: ColorValue):
        with qt_signals_blocked(self.color_edit):
            for layer in self._layers:
                getattr(
                    layer, self._colormanager_attribute
                ).current_color = color

    def _on_current_color_change(self):
        with qt_signals_blocked(self.color_edit):
            color = _unique_element(
                [
                    getattr(layer, self._colormanager_attribute).current_color
                    for layer in self._layers
                ]
            )
            self.color_edit.setColor(color)

    def change_color_feature(self, feature: str):
        if feature == _NONE_STRING:
            return
        with qt_signals_blocked(self.color_feature_combobox):
            for layer in self._layers:
                layer.color = feature
            self._on_color_mode_change()

    def _on_color_feature_changed(self):
        with qt_signals_blocked(self.color_mode_combobox):
            managers = [
                getattr(layer, self._colormanager_attribute)
                for layer in self._layers
            ]
            feature = _unique_element(
                [
                    manager.color_properties.name
                    if manager.color_properties is not None
                    else None
                    for manager in managers
                ]
            )
            if feature is not None:
                self.color_feature_combobox.setCurrentText(feature)
            else:
                self.color_feature_combobox.setCurrentText(_NONE_STRING)

    def _on_layer_features_change(self):
        with qt_signals_blocked(self.color_feature_combobox):
            prev = self.color_feature_combobox.currentText()
            common_features = set.intersection(
                *[
                    {str(c) for c in layer.features.columns}
                    for layer in self._layers
                ]
            )
            self.color_feature_combobox.clear()
            self.color_feature_combobox.addItem(_NONE_STRING)
            self.color_feature_combobox.addItems(common_features)
            if prev in common_features:
                self.color_feature_combobox.setCurrentText(prev)
        new = self.color_feature_combobox.currentText()
        if new != prev:
            # TODO: does it emit?
            self.color_feature_combobox.setCurrentText(new)

    def _update_visible_widgets(self, mode: ColorMode | None):
        # also accounts for mode = None in the mismatched layers case
        feature_based = mode in (ColorMode.cycle, ColorMode.colormap)
        synced = mode is not None
        self.color_label.setVisible(synced and not feature_based)
        self.color_edit.setVisible(synced and not feature_based)
        self.feature_label.setVisible(synced and feature_based)
        self.color_feature_combobox.setVisible(synced and feature_based)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.color_mode_label, self.color_mode_combobox),
            (self.color_label, self.color_edit),
            (self.feature_label, self.color_feature_combobox),
        ]
