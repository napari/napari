from qtpy.QtWidgets import QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari.layers import Points
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtBorderColorControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current border
    color layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    border_color_edit : napari._qt.widgets.qt_color_swatch.QColorSwatchEdit
        ColorSwatchEdit controlling current face color of the layer.
    border_color_edit_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current egde color chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Points) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.current_border_color.connect(
            self._on_current_border_color_change
        )
        self._layer._border.events.current_color.connect(
            self._on_current_border_color_change
        )

        # Setup widgets
        self.border_color_edit = QColorSwatchEdit(
            initial_color=self._layer.current_border_color,
            tooltip=trans._(
                'Click to set the border color of currently selected points and any added afterwards.'
            ),
        )
        connect_setattr(
            self.border_color_edit.color_changed,
            self._layer,
            'current_border_color',
        )

        self.border_color_edit_label = QtWrappedLabel(trans._('border color:'))

    def _on_current_border_color_change(self) -> None:
        """Receive layer.current_border_color() change event and update view."""
        with qt_signals_blocked(self.border_color_edit):
            self.border_color_edit.setColor(self._layer.current_border_color)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.border_color_edit_label, self.border_color_edit)]
