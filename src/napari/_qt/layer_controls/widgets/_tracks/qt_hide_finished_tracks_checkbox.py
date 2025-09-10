from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QWidget,
)

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers import Tracks


class QtHideFinishedTracksCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the hide finished
    tracks attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    hide_finished_tracks_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox controlling if finished tracks should be hidden.
    hide_finished_tracks_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for showing the option checkbox.
    """

    def __init__(self, parent: QWidget, layer: 'Tracks') -> None:
        super().__init__(parent, layer)
        # Setup layer

        # Setup widgets
        self.hide_finished_tracks_checkbox = QCheckBox()
        self.hide_finished_tracks_checkbox.stateChanged.connect(
            self.change_hide_finished_tracks
        )

        self.hide_finished_tracks_checkbox_label = QtWrappedLabel(
            trans._('hide completed:')
        )

    def change_hide_finished_tracks(self, state: int) -> None:
        self._layer.hide_finished_tracks = (
            self.hide_finished_tracks_checkbox.isChecked()
        )

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [
            (
                self.hide_finished_tracks_checkbox_label,
                self.hide_finished_tracks_checkbox,
            )
        ]
