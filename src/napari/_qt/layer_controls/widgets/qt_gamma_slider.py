from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtGammaSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current brush
    size attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    gamma_slider : superqt.QLabeledDoubleSlider
        Gamma adjustment slider widget.
    gamma_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the gamma chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.gamma.connect(self._on_gamma_change)

        # Setup widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self._layer.gamma)
        connect_setattr(sld.valueChanged, self._layer, 'gamma')
        self.gamma_slider = sld

        self.gamma_slider_label = QtWrappedLabel(trans._('gamma:'))

    def _on_gamma_change(self):
        """Receive the layer model gamma change event and update the slider."""
        with qt_signals_blocked(self.gamma_slider):
            self.gamma_slider.setValue(self._layer.gamma)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.gamma_slider_label, self.gamma_slider)]
