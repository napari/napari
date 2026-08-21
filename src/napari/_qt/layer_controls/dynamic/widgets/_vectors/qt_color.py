from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.utils._color_manager_constants import ColorMode
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from napari.layers import Vectors


class QtEdgeColorFeatureControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    color, color mode and color feature selection from the layer attributes and
    Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Vectors]
        A list of napari Vectors layers.

    Attributes
    ----------
    color_mode_combobox : qtpy.QtWidgets.QComboBox
        Dropdown to select the edge color mode.
    color_mode_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current selected color_mode chooser widget.
    color_edit : qtpy.QtWidgets.QSlider
        ColorSwatchEdit controlling current edge color of the layer.
    color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    color_feature_box : qtpy.QtWidgets.QComboBox
        Dropdown to select the feature for mapping color.
    feature_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color_feature_box chooser widget.
    """

    _layers: list[Vectors]

    def __init__(
        self, layers: list[Vectors], parent: QWidget | None = None
    ) -> None:
        super().__init__(layers, parent)
        # Setup layer
        for layer in self._layers:
            layer.events.color_mode.connect(self._on_color_mode_change)
            layer.events.color.connect(self._on_color_change)

        # dropdown to select the edge color mode
        self.color_mode_label = QtWrappedLabel('edge color mode:')
        self.color_mode_combobox = QComboBox(parent)
        color_modes = [e.value for e in ColorMode]
        self.color_mode_combobox.addItems(color_modes)
        self.color_mode_combobox.setCurrentText(self._layers[0].color_mode)
        self.color_mode_combobox.currentTextChanged.connect(
            self.change_color_mode
        )

        # dropdown to select the feature for mapping color
        self.feature_label = QtWrappedLabel('edge feature:')
        self.color_feature_box = QComboBox(parent)
        self.color_feature_box.addItems(self._layers[0].features.columns)
        if self._layers[0]._color_manager.color_properties is not None:
            self.color_feature_box.setCurrentText(
                self._layers[0]._color_manager.color_properties.name
            )
        self.color_feature_box.currentTextChanged.connect(
            self.change_color_feature
        )

        # vector direct color mode adjustment and widget
        self.color_edit = QColorSwatchEdit(
            initial_color=self._layers[0].color,
            tooltip='Click to set current edge color',
        )
        for layer in self._layers:
            connect_setattr(self.color_edit.color_changed, layer, 'color')
        self.color_label = QtWrappedLabel('edge color:')

    def change_color_feature(self, feature: str):
        """Change color feature of vectors on the layer model.

        Parameters
        ----------
        feature : str
            feature to map the edge color to
        """
        with qt_signals_blocked(self.color_feature_box):
            for layer in self._layers:
                layer.color = feature
            self._on_color_mode_change()

    def change_color_mode(self, mode: str):
        """Change edge color mode of vectors on the layer model.

        Parameters
        ----------
        mode : str
            Edge color for vectors. Must be: 'direct', 'cycle', or 'colormap'
        """
        with qt_signals_blocked(self.color_mode_combobox):
            for layer in self._layers:
                layer.color_mode = mode
            self._update_color_gui(mode)

    def _on_color_mode_change(self):
        """Receive layer model edge color mode change event & update dropdown."""
        with qt_signals_blocked(self.color_mode_combobox):
            mode = self._layers[0]._color_manager.color_mode
            index = self.color_mode_combobox.findText(
                mode, Qt.MatchFixedString
            )
            self.color_mode_combobox.setCurrentIndex(index)

            self._update_color_gui(mode)

    def _on_color_change(self):
        """Receive layer model edge color  change event & update dropdown."""
        if (
            self._layers[0]._color_manager.color_mode == ColorMode.DIRECT
            and len(self._layers[0].data) > 0
        ):
            with qt_signals_blocked(self.color_edit):
                self.color_edit.setColor(self._layers[0].color[0])
        elif self._layers[0]._color_manager.color_mode in (
            ColorMode.CYCLE,
            ColorMode.COLORMAP,
        ):
            with qt_signals_blocked(self.color_feature_box):
                prop = self._layers[0]._color_manager.color_properties.name
                index = self.color_feature_box.findText(
                    prop, Qt.MatchFixedString
                )
                self.color_feature_box.setCurrentIndex(index)

    def _update_color_gui(self, mode: str):
        """Update the GUI element associated with color.
        This is typically used when color_mode changes

        Parameters
        ----------
        mode : str
            The new color mode the GUI needs to be updated for.
            Should be: 'direct', 'cycle', 'colormap'
        """
        if mode in {'cycle', 'colormap'}:
            self.color_edit.setHidden(True)
            self.color_label.setHidden(True)
            self.color_feature_box.setHidden(False)
            self.feature_label.setHidden(False)

        elif mode == 'direct':
            self.color_edit.setHidden(False)
            self.color_label.setHidden(False)
            self.color_feature_box.setHidden(True)
            self.feature_label.setHidden(True)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.color_mode_label, self.color_mode_combobox),
            (self.color_label, self.color_edit),
            (self.feature_label, self.color_feature_box),
        ]
