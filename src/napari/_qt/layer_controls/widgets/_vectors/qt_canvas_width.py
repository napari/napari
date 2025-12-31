from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtFixedCanvasWidthCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer fixed canvas
    width attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    fixed_canvas_width_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to fix the canvas width.
    fixed_canvas_width_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the fixed canvas width enablement chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.fixed_canvas_width.connect(
            self._on_fixed_canvas_width_change
        )

        # Setup widgets
        self.fixed_canvas_width_checkbox = QCheckBox()
        self.fixed_canvas_width_checkbox.setToolTip(
            trans._('Fixed canvas width')
        )
        self.fixed_canvas_width_checkbox.setChecked(
            self._layer.fixed_canvas_width
        )
        connect_setattr(
            self.fixed_canvas_width_checkbox.stateChanged,
            layer,
            'fixed_canvas_width',
            convert_fun=checked_to_bool,
        )

        self.fixed_canvas_width_checkbox_label = QtWrappedLabel(
            trans._('fixed canvas width:')
        )

    def _on_fixed_canvas_width_change(self) -> None:
        """Receive layer model fixed_canvas_width change event and update checkbox."""
        with qt_signals_blocked(self.fixed_canvas_width_checkbox):
            self.fixed_canvas_width_checkbox.setChecked(
                self._layer.fixed_canvas_width
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.fixed_canvas_width_checkbox_label,
                self.fixed_canvas_width_checkbox,
            )
        ]
