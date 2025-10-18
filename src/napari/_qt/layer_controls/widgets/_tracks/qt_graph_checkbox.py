from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool
from napari.layers import Tracks
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtGraphCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the graph should be
    displayed attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    graph_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if graph of the layer should be shown.
    graph_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the graph chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        # NOTE(arl): there are no events fired for changing checkbox (layer `display_graph` attribute)

        # Setup widgets
        self.graph_checkbox = QCheckBox()
        self.graph_checkbox.setChecked(True)
        connect_setattr(
            self.graph_checkbox.stateChanged,
            layer,
            'display_graph',
            convert_fun=checked_to_bool,
        )

        self.graph_checkbox_label = QtWrappedLabel(trans._('graph:'))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.graph_checkbox_label, self.graph_checkbox)]
