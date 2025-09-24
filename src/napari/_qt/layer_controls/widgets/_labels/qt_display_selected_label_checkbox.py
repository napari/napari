from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.layers import Labels
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtDisplaySelectedLabelCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer attribute to
    only display selected label and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    selected_color_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if only currently selected label is shown.
    selected_color_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer should show only currently selected label chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.show_selected_label.connect(
            self._on_show_selected_label_change
        )

        # Setup widgets
        selected_color_checkbox = QCheckBox()
        selected_color_checkbox.setToolTip(
            trans._('Display only selected label')
        )
        connect_setattr(
            selected_color_checkbox.stateChanged,
            layer,
            'show_selected_label',
            convert_fun=checked_to_bool,
        )
        self.selected_color_checkbox = selected_color_checkbox
        self._on_show_selected_label_change()

        self.selected_color_checkbox_label = QtWrappedLabel(
            trans._('show\nselected:')
        )

    def _on_show_selected_label_change(self) -> None:
        """Receive layer model show_selected_labels event and update the checkbox."""
        with qt_signals_blocked(self.selected_color_checkbox):
            self.selected_color_checkbox.setChecked(
                self._layer.show_selected_label
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (self.selected_color_checkbox_label, self.selected_color_checkbox)
        ]
