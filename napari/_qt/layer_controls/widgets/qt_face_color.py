import numpy as np
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QLabel, QWidget

from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtFaceColorControl(QObject):
    def __init__(self, parent: QWidget, layer: Layer):
        super().__init__(parent)
        # Setup layer
        self.layer = layer
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )

        # Setup widgets
        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self.faceColorLabel = QLabel(trans._('face color:'))
        self._on_current_face_color_change()
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)

    def changeFaceColor(self, color: np.ndarray):
        """Change face color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Face color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    def _on_current_face_color_change(self):
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def get_widget_controls(self):
        """
        Enable access to the created labels and control widgets.

        Returns
        -------
        list
            List of tuples of the label and widget controls available.

        """
        return [(self.faceColorLabel, self.faceColorEdit)]
