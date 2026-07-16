from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtGammaSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current gamma
    attribute value and Qt widgets.

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
        # Setup widgets
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self._layer.gamma)
        connect_setattr(sld.valueChanged, self._layer, 'gamma')
        self._callbacks.append(
            attr_to_settr(self._layer, 'gamma', sld, 'setValue')
        )
        self.gamma_slider = sld

        self.gamma_slider_label = QtWrappedLabel(trans._('gamma:'))

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.gamma_slider_label, self.gamma_slider)]
