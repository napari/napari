from collections.abc import Iterable

import numpy as np
from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QLabel, QWidget

from napari._qt.widgets._slider_compat import QSlider
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtEdgeWidthSliderControl(QObject):
    def __init__(self, parent: QWidget, layer: Layer):
        super().__init__(parent)
        # Setup layer
        self.layer = layer
        self.layer.events.edge_width.connect(self._on_edge_width_change)

        # Setup widgets
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.edgeWidthSlider = sld
        self.edgeWidthLabel = QLabel(trans._('edge width:'))

    def changeWidth(self, value):
        """Change edge line width of shapes on the layer model.

        Parameters
        ----------
        value : float
            Line width of shapes.
        """
        self.layer.current_edge_width = float(value)

    def _on_edge_width_change(self):
        """Receive layer model edge line width change event and update slider."""
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(value), 0, 40)
            self.edgeWidthSlider.setValue(value)

    def get_widget_controls(self):
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list
            List of tuples of the label and widget controls available.

        """
        return [(self.edgeWidthLabel, self.edgeWidthSlider)]
