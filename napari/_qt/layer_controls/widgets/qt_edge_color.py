import numpy as np
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QWidget

from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtEdgeColorControl(QObject):
    def __init__(self, parent: QWidget, layer: Layer):
        super().__init__(parent)
        # Setup layer
        self.layer = layer
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )

        # Setup widgets
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._('click to set current edge color'),
        )
        self.edgeColorLabel = trans._('edge color:')
        self._on_current_edge_color_change()
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

    def changeEdgeColor(self, color: np.ndarray):
        """Change edge color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def _on_current_edge_color_change(self):
        """Receive layer model edge color change event and update color swatch."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def get_widget_controls(self):
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list
            List of tuples of the label and widget controls available.

        """
        return [(self.edgeColorLabel, self.edgeColorEdit)]
