from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.layers.utils._color_manager_constants import ColorMode
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtEdgeColorPropertyControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    color, color mode and color property selection from the layer attributes and
    Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    color_mode_combobox : qtpy.QtWidgets.QComboBox
        Dropdown to select the edge color mode.
    color_mode_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current selected edge_color_mode chooser widget.
    edge_color_edit : qtpy.QtWidgets.QSlider
        ColorSwatchEdit controlling current edge color of the layer.
    edge_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    color_prop_box : qtpy.QtWidgets.QComboBox
        Dropdown to select the property for mapping edge_color.
    edge_prop_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current selected _edge_color_property chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.edge_color_mode.connect(
            self._on_edge_color_mode_change
        )
        self._layer.events.edge_color.connect(self._on_edge_color_change)

        # Setup widgets
        # dropdown to select the property for mapping edge_color
        color_properties = self._get_property_values()
        self.color_prop_box = QComboBox(parent)
        self.color_prop_box.currentTextChanged.connect(
            self.change_edge_color_property
        )
        self.color_prop_box.addItems(color_properties)
        self.edge_prop_label = QtWrappedLabel(trans._('edge property:'))

        # vector direct color mode adjustment and widget
        self.edge_color_edit = QColorSwatchEdit(
            initial_color=self._layer.edge_color,
            tooltip=trans._(
                'Click to set current edge color',
            ),
        )
        connect_setattr(
            self.edge_color_edit.color_changed, self._layer, 'edge_color'
        )
        self.edge_color_label = QtWrappedLabel(trans._('edge color:'))
        self._on_edge_color_change()

        # dropdown to select the edge color mode
        self.color_mode_combobox = QComboBox(parent)
        color_modes = [e.value for e in ColorMode]
        self.color_mode_combobox.addItems(color_modes)
        self.color_mode_combobox.currentTextChanged.connect(
            self.change_edge_color_mode
        )
        self.color_mode_label = QtWrappedLabel(trans._('edge color mode:'))
        self._on_edge_color_mode_change()

    def change_edge_color_property(self, property_name: str):
        """Change edge_color_property of vectors on the layer model.
        This property is the property the edge color is mapped to.

        Parameters
        ----------
        property_name : str
            property to map the edge color to
        """
        mode = self._layer.edge_color_mode
        try:
            self._layer.edge_color = property_name
            self._layer.edge_color_mode = mode
        except TypeError:
            # if the selected property is the wrong type for the current color mode
            # the color mode will be changed to the appropriate type, so we must update
            self._on_edge_color_mode_change()
            raise

    def change_edge_color_mode(self, mode: str):
        """Change edge color mode of vectors on the layer model.

        Parameters
        ----------
        mode : str
            Edge color for vectors. Must be: 'direct', 'cycle', or 'colormap'
        """
        old_mode = self._layer.edge_color_mode
        with self._layer.events.edge_color_mode.blocker():
            try:
                self._layer.edge_color_mode = mode
                self._update_edge_color_gui(mode)

            except ValueError:
                # if the color mode was invalid, revert to the old mode (layer and GUI)
                self._layer.edge_color_mode = old_mode
                self.color_mode_combobox.setCurrentText(old_mode)
                raise

    def _on_edge_color_mode_change(self):
        """Receive layer model edge color mode change event & update dropdown."""
        if not hasattr(self, 'color_mode_combobox'):
            # Ignore early events i.e when widgets haven't been created yet.
            return

        with qt_signals_blocked(self.color_mode_combobox):
            mode = self._layer._edge.color_mode
            index = self.color_mode_combobox.findText(
                mode, Qt.MatchFixedString
            )
            self.color_mode_combobox.setCurrentIndex(index)

            self._update_edge_color_gui(mode)

    def _on_edge_color_change(self):
        """Receive layer model edge color  change event & update dropdown."""
        if (
            self._layer._edge.color_mode == ColorMode.DIRECT
            and len(self._layer.data) > 0
        ):
            with qt_signals_blocked(self.edge_color_edit):
                self.edge_color_edit.setColor(self._layer.edge_color[0])
        elif self._layer._edge.color_mode in (
            ColorMode.CYCLE,
            ColorMode.COLORMAP,
        ):
            with qt_signals_blocked(self.color_prop_box):
                prop = self._layer._edge.color_properties.name
                index = self.color_prop_box.findText(prop, Qt.MatchFixedString)
                self.color_prop_box.setCurrentIndex(index)

    def _get_property_values(self):
        """Get the current property values from the Vectors layer

        Returns
        -------
        property_values : np.ndarray
            array of all of the union of the property names (keys)
            in Vectors.properties and Vectors.property_choices

        """
        property_choices = [*self._layer.property_choices]
        properties = [*self._layer.properties]
        property_values = np.union1d(property_choices, properties)

        return property_values

    def _update_edge_color_gui(self, mode: str):
        """Update the GUI element associated with edge_color.
        This is typically used when edge_color_mode changes

        Parameters
        ----------
        mode : str
            The new edge_color mode the GUI needs to be updated for.
            Should be: 'direct', 'cycle', 'colormap'
        """
        if mode in {'cycle', 'colormap'}:
            self.edge_color_edit.setHidden(True)
            self.edge_color_label.setHidden(True)
            self.color_prop_box.setHidden(False)
            self.edge_prop_label.setHidden(False)

        elif mode == 'direct':
            self.edge_color_edit.setHidden(False)
            self.edge_color_label.setHidden(False)
            self.color_prop_box.setHidden(True)
            self.edge_prop_label.setHidden(True)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.color_mode_label, self.color_mode_combobox),
            (self.edge_color_label, self.edge_color_edit),
            (self.edge_prop_label, self.color_prop_box),
        ]
