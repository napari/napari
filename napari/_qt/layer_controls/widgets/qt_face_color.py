import numpy as np
from qtpy.QtWidgets import QLabel, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers.base.base import Layer
from napari.utils.translations import trans


class QtFaceColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current face
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        faceColorEdit : qtpy.QtWidgets.QSlider
            ColorSwatchEdit controlling current face color of the layer.
        faceColorLabel : qtpy.QtWidgets.QLabel
            Label for the current face color widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )

        # Setup widgets
        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self._layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self.faceColorLabel = QLabel(trans._('face color:'))
        self._on_current_face_color_change()
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)

    def changeFaceColor(self, color: np.ndarray) -> None:
        """Change face color of shapes.

        Parameters
        ----------
        color : np.ndarray
            Face color for shapes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self._layer.events.current_face_color.blocker():
            self._layer.current_face_color = color

    def _on_current_face_color_change(self) -> None:
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self._layer.current_face_color)

    def get_widget_controls(self) -> list[tuple[QLabel, QWidget]]:
        return [(self.faceColorLabel, self.faceColorEdit)]
