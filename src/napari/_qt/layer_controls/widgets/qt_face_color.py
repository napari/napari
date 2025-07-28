from typing import Optional

from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers import Points, Shapes
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtFaceColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current face
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Shapes | napari.layers.Points
        An instance of a napari Shapes or Points layer.
    toolip : str
        String to use for the tooltip of the face color edit widget.

    Attributes
    ----------
    face_color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current face color of the layer.
    face_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current face color chooser widget.
    """

    def __init__(
        self,
        parent: QWidget,
        layer: Shapes | Points,
        tooltip: Optional[str] = None,
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        if hasattr(self._layer, '_face'):
            # Handle Points layer case
            self._layer._face.events.current_color.connect(
                self._on_current_face_color_change
            )

        # Setup widgets
        self.face_color_edit = QColorSwatchEdit(
            initial_color=self._layer.current_face_color,
            tooltip=tooltip,
        )
        self.face_color_label = QtWrappedLabel(trans._('face color:'))
        self._on_current_face_color_change()
        connect_setattr(
            self.face_color_edit.color_changed,
            self._layer,
            'current_face_color',
        )

    def _on_current_face_color_change(self) -> None:
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.face_color_edit):
            self.face_color_edit.setColor(self._layer.current_face_color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.face_color_label, self.face_color_edit)]
