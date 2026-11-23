from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Tracks
from napari.utils.events.event_utils import connect_setattr


class QtFadeToSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current fade_to
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Tracks
        An instance of a napari Tracks layer.

    Attributes
    ----------
    fade_to_slider : superqt.QLabeledDoubleSlider
        Slider controlling fade_to of the layer.
    fade_to_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the fade_to chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Tracks) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.fade_to.connect(self._on_fade_to_change)

        # Setup widgets
        self.fade_to_slider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, parent=parent
        )
        self.fade_to_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fade_to_slider.setMinimum(0.0)
        self.fade_to_slider.setMaximum(1.0)
        self.fade_to_slider.setSingleStep(0.01)
        self.fade_to_slider.setValue(self._layer.fade_to)
        connect_setattr(
            self.fade_to_slider.valueChanged, self._layer, 'fade_to'
        )

        self.fade_to_slider_label = QtWrappedLabel('fade to:')

        self._on_fade_to_change()

    def _on_fade_to_change(self) -> None:
        """Receive layer model fade_to change event and update slider."""
        with qt_signals_blocked(self.fade_to_slider):
            self.fade_to_slider.setValue(self._layer.fade_to)

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QWidget] | tuple[QWidget]]:
        return [(self.fade_to_slider_label, self.fade_to_slider)]
