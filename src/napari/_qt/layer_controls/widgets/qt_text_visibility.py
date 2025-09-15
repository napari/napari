from qtpy.QtWidgets import QCheckBox, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.events import disconnect_events
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtTextVisibilityControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the text visibility
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    text_disp_checkbox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer is visible or not.
    text_disp_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the text visibility widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.text.events.visible.connect(
            self._on_text_visibility_change
        )

        # Setup widgets
        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('Toggle text visibility'))
        text_disp_cb.setChecked(self._layer.text.visible)
        connect_setattr(
            text_disp_cb.stateChanged,
            layer.text,
            'visible',
            convert_fun=checked_to_bool,
        )
        self.text_disp_checkbox = text_disp_cb
        self.text_disp_label = QtWrappedLabel(trans._('display text:'))

    def _on_text_visibility_change(self) -> None:
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with qt_signals_blocked(self.text_disp_checkbox):
            self.text_disp_checkbox.setChecked(self._layer.text.visible)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.text_disp_label, self.text_disp_checkbox)]

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self._layer.text.events, self)
        super().disconnect_widget_controls()
