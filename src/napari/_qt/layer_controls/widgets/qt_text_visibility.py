from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Points, Shapes
from napari.utils.events import disconnect_events
from napari.utils.events.event_utils import connect_setattr


class QtTextVisibilityControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the text visibility
    and text scaling layer attributes and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Points | napari.layers.Shapes
        An instance of a napari layer.

    Attributes
    ----------
    text_disp_checkbox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer is visible or not.
    text_scaling_checkbox : qtpy.QtWidgets.QCheckbox
        Checkbox controlling if text on the layer scales with the zoom level.
    text_widget : qtpy.QtWidgets.QWidget
        Widget holding the text visibility and text scaling checkboxes in a row.
    text_disp_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the text widget.
    """

    _layer: Points | Shapes

    def __init__(self, parent: QWidget, layer: Points | Shapes) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.text.events.visible.connect(
            self._on_text_visibility_change
        )
        self._layer.text.events.scaling.connect(self._on_text_scaling_change)

        # Setup widgets
        text_disp_cb = QCheckBox('display')
        text_disp_cb.setToolTip('Toggle text visibility')
        text_disp_cb.setChecked(self._layer.text.visible)
        connect_setattr(
            text_disp_cb.toggled,
            layer.text,
            'visible',
        )
        self.text_disp_checkbox = text_disp_cb

        text_scaling_cb = QCheckBox('scale')
        text_scaling_cb.setToolTip('Toggle text scaling with zoom level')
        text_scaling_cb.setChecked(self._layer.text.scaling)
        connect_setattr(
            text_scaling_cb.toggled,
            layer.text,
            'scaling',
        )
        self.text_scaling_checkbox = text_scaling_cb

        text_layout = QHBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(self.text_disp_checkbox)
        text_layout.addWidget(self.text_scaling_checkbox)
        text_layout.addStretch(1)
        self.text_widget = QWidget()
        self.text_widget.setLayout(text_layout)
        self.text_widget.setStyleSheet('background: transparent;')

        self.text_disp_label = QtWrappedLabel('text:')

    def _on_text_visibility_change(self) -> None:
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with qt_signals_blocked(self.text_disp_checkbox):
            self.text_disp_checkbox.setChecked(self._layer.text.visible)

    def _on_text_scaling_change(self) -> None:
        """Receive layer model text scaling change event and update checkbox."""
        with qt_signals_blocked(self.text_scaling_checkbox):
            self.text_scaling_checkbox.setChecked(self._layer.text.scaling)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.text_disp_label, self.text_widget)]

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self._layer.text.events, self)
        super().disconnect_widget_controls()
