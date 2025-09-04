from typing import Optional

from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers import Shapes
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtEdgeColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current edge
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Shapes
        An instance of a napari Shapes layer.
    toolip : str
        String to use for the tooltip of the edge color edit widget.

    Attributes
    ----------
    edge_color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current edge color of the layer.
    edge_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Shapes, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )

        # Setup widgets
        self.edge_color_edit = QColorSwatchEdit(
            initial_color=self._layer.current_edge_color,
            tooltip=tooltip,
        )
        self._on_current_edge_color_change()
        connect_setattr(
            self.edge_color_edit.color_changed,
            self._layer,
            'current_edge_color',
        )
        self.edge_color_label = QtWrappedLabel(trans._('edge color:'))

    def _on_current_edge_color_change(self) -> None:
        """Receive layer model edge color change event and update color swatch."""
        with qt_signals_blocked(self.edge_color_edit):
            self.edge_color_edit.setColor(self._layer.current_edge_color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.edge_color_label, self.edge_color_edit)]
