from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
    _QtABCMeta,
)
from napari._qt.utils import attr_to_settr
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Shapes


class QtEdgeColorControl(QtWidgetControlsBase, metaclass=_QtABCMeta):
    """
    Class that wraps the connection of events/signals between the current edge
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Shapes]
        A list of napari Shapes layers.
    toolip : str
        String to use for the tooltip of the edge color edit widget.

    Attributes
    ----------
    edge_color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current edge color of the layer.
    edge_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    """

    _layer: Shapes

    def __init__(
        self,
        parent: QWidget,
        layers: list[Shapes],
        tooltip: Optional[str] = None,
    ) -> None:
        super().__init__(parent, layers)
        self._layers = layers
        # Setup widgets
        self.edge_color_edit = QColorSwatchEdit(
            initial_color=self._layers[0].current_edge_color,
            tooltip=tooltip,
        )
        for layer in self._layers:
            connect_setattr(
                self.edge_color_edit.color_changed,
                layer,
                'current_edge_color',
            )
        self._callbacks.append(  # @lorenzo: does this trigger the widget to change? in that case I only do it for the first layer no? -> yes, @margot check other files and change there
            attr_to_settr(
                self._layers[0],
                'current_edge_color',
                self.edge_color_edit,
                'setColor',
            )
        )
        self.edge_color_label = QtWrappedLabel('edge color:')

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.edge_color_label, self.edge_color_edit)]
