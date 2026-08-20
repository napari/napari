from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Points, Shapes


class QtFaceColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current face
    color layer attribute and Qt widgets.

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
    face_color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current face color of the layer.
    face_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current face color chooser widget.
    """

    _layers: list[Points | Shapes]

    def __init__(
        self,
        layers: list[Points | Shapes],
        parent: QWidget | None = None,
        tooltip: Optional[str] = None,
    ) -> None:
        super().__init__(layers, parent)
        # Setup widgets
        self.face_color_edit = QColorSwatchEdit(
            initial_color=self._layers[0].current_face_color,
            tooltip=tooltip,
        )
        self.face_color_label = QtWrappedLabel('face color:')
        for layer in self._layers:
            connect_setattr(
                self.face_color_edit.color_changed,
                layer,
                'current_face_color',
            )
        self._callbacks.append(
            attr_to_settr(
                self._layers[0],
                'current_face_color',
                self.face_color_edit,
                'setColor',
            )
        )
        if hasattr(self._layers[0], '_face'):
            # Handle Points layer case
            self._callbacks.append(
                attr_to_settr(
                    self._layers[0]._face,
                    'current_color',
                    self.face_color_edit,
                    'setColor',
                )
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.face_color_label, self.face_color_edit)]
