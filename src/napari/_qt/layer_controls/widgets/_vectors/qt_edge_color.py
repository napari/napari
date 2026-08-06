import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget
from superqt import QLabeledDoubleRangeSlider

from napari._qt.layer_controls.widgets.qt_colormap_control import (
    QtColormapComboBox,
)
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers import Vectors
from napari.layers.utils._color_manager_constants import ColorMode
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events.event_utils import connect_setattr


class QtEdgeColorFeatureControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    color, color mode and color feature selection from the layer attributes and
    Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

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
    color_feature_box : qtpy.QtWidgets.QComboBox
        Dropdown to select the feature for mapping edge_color.
    edge_feature_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the color_feature_box chooser widget.
    colormap_combobox : qtpy.QtWidgets.QComboBox
        Dropdown to select the colormap used in colormap mode.
    colormap_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the colormap_combobox chooser widget.
    contrast_limits_slider : superqt.QLabeledDoubleRangeSlider
        Slider controlling the feature values mapped to the ends of the colormap.
    contrast_limits_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the contrast_limits_slider widget.
    """

    def __init__(self, parent: QWidget, layer: Vectors) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.edge_color_mode.connect(
            self._on_edge_color_mode_change
        )
        self._layer.events.edge_color.connect(self._on_edge_color_change)
        self._layer.events.edge_colormap.connect(self._on_colormap_change)
        self._layer.events.edge_contrast_limits.connect(
            self._on_contrast_limits_change
        )

        # Setup widgets
        # dropdown to select the feature for mapping edge_color
        self.color_feature_box = QComboBox(parent)
        self.color_feature_box.currentTextChanged.connect(
            self.change_edge_color_feature
        )
        self.color_feature_box.addItems(self._layer.features.columns)
        self.edge_feature_label = QtWrappedLabel('edge feature:')

        # The mapping half of colormap mode: which colormap, over which values.
        self.colormap_combobox = QtColormapComboBox(parent)
        self.colormap_combobox.setObjectName('colormapComboBox')
        for name, colormap in AVAILABLE_COLORMAPS.items():
            self.colormap_combobox.addItem(colormap._display_name, name)
        self.colormap_combobox.currentTextChanged.connect(self.change_colormap)
        self.colormap_label = QtWrappedLabel('edge colormap:')

        self.contrast_limits_slider = QLabeledDoubleRangeSlider(
            Qt.Orientation.Horizontal, parent
        )
        self.contrast_limits_slider.setMinimum(0)
        self.contrast_limits_slider.setMaximum(1)
        self.contrast_limits_slider.valueChanged.connect(
            self.change_contrast_limits
        )
        self.contrast_limits_label = QtWrappedLabel('edge contrast limits:')
        self._on_colormap_change()
        self._on_contrast_limits_change()

        # vector direct color mode adjustment and widget
        self.edge_color_edit = QColorSwatchEdit(
            initial_color=self._layer.edge_color,
            tooltip='Click to set current edge color',
        )
        connect_setattr(
            self.edge_color_edit.color_changed, self._layer, 'edge_color'
        )
        self.edge_color_label = QtWrappedLabel('edge color:')
        self._on_edge_color_change()

        # dropdown to select the edge color mode
        self.color_mode_combobox = QComboBox(parent)
        color_modes = [e.value for e in ColorMode]
        self.color_mode_combobox.addItems(color_modes)
        self.color_mode_combobox.currentTextChanged.connect(
            self.change_edge_color_mode
        )
        self.color_mode_label = QtWrappedLabel('edge color mode:')
        self._on_edge_color_mode_change()

    def change_edge_color_feature(self, feature: str):
        """Change edge_color feature of vectors on the layer model.

        Parameters
        ----------
        feature : str
            feature to map the edge color to
        """
        mode = self._layer.edge_color_mode
        try:
            self._layer.edge_color = feature
            self._layer.edge_color_mode = mode
        except TypeError:
            # if the selected feature is the wrong type for the current color mode
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

    def change_colormap(self, colormap: str):
        """Change the colormap the mapped feature is rendered through.

        Parameters
        ----------
        colormap : str
            Display name of the colormap, as shown in the dropdown.
        """
        self._layer.edge_colormap = self.colormap_combobox.currentData()

    def change_contrast_limits(self, contrast_limits: tuple[float, float]):
        """Change the feature values mapped to the ends of the colormap.

        Parameters
        ----------
        contrast_limits : tuple of float
            The low and high feature value.
        """
        self._layer.edge_contrast_limits = tuple(contrast_limits)

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update the dropdown."""
        if not hasattr(self, 'colormap_combobox'):
            # Ignore early events i.e when widgets haven't been created yet.
            return
        name = self._layer.edge_colormap.name
        index = self.colormap_combobox.findData(name)
        if index != -1 and index != self.colormap_combobox.currentIndex():
            with qt_signals_blocked(self.colormap_combobox):
                self.colormap_combobox.setCurrentIndex(index)

    def _on_contrast_limits_change(self):
        """Receive layer model contrast limits change event and update the slider.

        The slider spans the mapped feature's values, widened to reach limits set
        outside them.
        """
        if not hasattr(self, 'contrast_limits_slider'):
            # Ignore early events i.e when widgets haven't been created yet.
            return
        limits = self._layer.edge_contrast_limits
        if limits is None:
            return
        low, high = float(limits[0]), float(limits[1])
        values = self._feature_values()
        if values is not None and len(values):
            low = min(low, float(np.min(values)))
            high = max(high, float(np.max(values)))
        if high <= low:
            high = low + 1.0
        with qt_signals_blocked(self.contrast_limits_slider):
            self.contrast_limits_slider.setRange(low, high)
            self.contrast_limits_slider.setValue(tuple(limits))

    def _feature_values(self):
        """Values of the feature currently mapped to color, or None."""
        color_properties = self._layer._edge.color_properties
        if color_properties is None:
            return None
        return np.asarray(color_properties.values, dtype=float)

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
            with qt_signals_blocked(self.color_feature_box):
                prop = self._layer._edge.color_properties.name
                index = self.color_feature_box.findText(
                    prop, Qt.MatchFixedString
                )
                self.color_feature_box.setCurrentIndex(index)

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
            self.color_feature_box.setHidden(False)
            self.edge_feature_label.setHidden(False)

        elif mode == 'direct':
            self.edge_color_edit.setHidden(False)
            self.edge_color_label.setHidden(False)
            self.color_feature_box.setHidden(True)
            self.edge_feature_label.setHidden(True)

        # Only colormap mode has a colormap to choose and limits to map through.
        colormapped = mode == 'colormap'
        self.colormap_combobox.setHidden(not colormapped)
        self.colormap_label.setHidden(not colormapped)
        self.contrast_limits_slider.setHidden(not colormapped)
        self.contrast_limits_label.setHidden(not colormapped)
        if colormapped:
            self._on_colormap_change()
            self._on_contrast_limits_change()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.color_mode_label, self.color_mode_combobox),
            (self.edge_color_label, self.edge_color_edit),
            (self.edge_feature_label, self.color_feature_box),
            (self.colormap_label, self.colormap_combobox),
            (self.contrast_limits_label, self.contrast_limits_slider),
        ]
