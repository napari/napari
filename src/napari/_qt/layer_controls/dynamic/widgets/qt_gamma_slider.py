from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.utils.events.event_utils import connect_setattr

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari.layers import Image, Surface


class QtGammaSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current gamma
    attribute value and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layers : list[napari.layers.Image | napari.layers.Surface]
        A list of Image and Surface napari layers.

    Attributes
    ----------
    gamma_slider : superqt.QLabeledDoubleSlider
        Gamma adjustment slider widget.
    gamma_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the gamma chooser widget.
    """

    def __init__(self, parent: QWidget, layers: list[Image | Surface]) -> None:
        super().__init__(parent, layers)

        # Setup gamma slider - exactly like opacity slider
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self._layers[0].gamma)
        for layer in self._layers:
            connect_setattr(sld.valueChanged, layer, 'gamma')
            self._callbacks.append(
                attr_to_settr(layer, 'gamma', sld, 'setValue')
            )
        self.gamma_slider = sld

        self.gamma_slider_label = QtWrappedLabel('gamma:')

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.gamma_slider_label, self.gamma_slider)]
