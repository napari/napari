from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import checked_to_bool, qt_signals_blocked
from napari.layers import Tracks
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtHideCompletedTracksCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the hide completed
    tracks attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    hide_completed_tracks_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if completed tracks should be hidden.
    hide_completed_tracks_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the option checkbox.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer

        # Setup widgets
        self.hide_completed_tracks_checkbox = QCheckBox()
        connect_setattr(
            self.hide_completed_tracks_checkbox.stateChanged,
            layer,
            'hide_completed_tracks',
            convert_fun=checked_to_bool,
        )

        self.hide_completed_tracks_checkbox_label = QtWrappedLabel(
            trans._('hide completed:')
        )

    def _on_hide_completed_tracks_change(self) -> None:
        """Receive layer model hide_completed_tracks event and update the checkbox."""
        with qt_signals_blocked(self.hide_completed_tracks_checkbox):
            self.hide_completed_tracks_checkbox.setChecked(
                self._layer.hide_completed_tracks
            )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.hide_completed_tracks_checkbox_label,
                self.hide_completed_tracks_checkbox,
            )
        ]
