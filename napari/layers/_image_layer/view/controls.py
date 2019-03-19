from PyQt5.QtWidgets import QHBoxLayout, QWidget
from PyQt5.QtCore import QSize

from ....util.range_slider import QVRangeSlider


class QtImageControls(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.clim.connect(self.clim_slider_update)

        # Create clim slider
        self.climSlider = QVRangeSlider(slider_range=[0, 1, 0.0001],
                                        values=[0, 1], parent=self)
        self.climSlider.setEmitWhileMoving(True)
        self.climSlider.collapsable = False
        self.climSlider.setEnabled(True)

        layout = QHBoxLayout()
        layout.addWidget(self.climSlider)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.climSlider.rangeChanged.connect(self.clim_slider_changed)

    def clim_slider_changed(self, slidermin, slidermax):
        from ..model import Image
        for layer in self.layer.viewer.layers:
            if isinstance(layer, Image) and layer.selected:
                valmin, valmax = layer._clim_range
                cmin = valmin+slidermin*(valmax-valmin)
                cmax = valmin+slidermax*(valmax-valmin)
                layer.clim = [cmin, cmax]
                layer._qt_controls.clim_slider_update(None)

    def clim_slider_update(self, event):
        valmin, valmax = self.layer._clim_range
        cmin, cmax = self.layer.clim
        slidermin = (cmin - valmin)/(valmax - valmin)
        slidermax = (cmax - valmin)/(valmax - valmin)
        self.climSlider.blockSignals(True)
        self.climSlider.setValues((slidermin, slidermax))
        self.climSlider.blockSignals(False)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._clim_msg
