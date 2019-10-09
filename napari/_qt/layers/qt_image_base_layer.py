from qtpy.QtWidgets import QHBoxLayout
from .. import QHRangeSlider
from .qt_base_layer import QtLayerControls
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox
from qtpy.QtGui import QImage, QPixmap


class QtBaseImageControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            lambda e: self.contrast_limits_slider_update()
        )

        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        comboBox._allitems = set(self.layer.colormaps)
        comboBox.activated[str].connect(
            lambda text=comboBox: self.changeColor(text)
        )
        self.colormapComboBox = comboBox

        # Create contrast_limits slider
        self.contrastLimitsSlider = QHRangeSlider(
            slider_range=[0, 1, 0.0001], values=[0, 1]
        )
        self.contrastLimitsSlider.setEmitWhileMoving(True)
        self.contrastLimitsSlider.collapsable = False
        self.contrastLimitsSlider.setEnabled(True)

        self.contrastLimitsSlider.rangeChanged.connect(
            self.contrast_limits_slider_changed
        )
        self.contrast_limits_slider_update()

        self.colorbarLabel = QLabel()
        self.colorbarLabel.setObjectName('colorbar')
        self.colorbarLabel.setToolTip('Colorbar')

        self._on_colormap_change(None)

    def changeColor(self, text):
        self.layer.colormap = text

    def _on_colormap_change(self, event):
        name = self.layer.colormap[0]
        if name not in self.colormapComboBox._allitems:
            self.colormapComboBox._allitems.add(name)
            self.colormapComboBox.addItem(name)
        if name != self.colormapComboBox.currentText():
            self.colormapComboBox.setCurrentText(name)

        # Note that QImage expects the image width followed by height
        image = QImage(
            self.layer._colorbar,
            self.layer._colorbar.shape[1],
            self.layer._colorbar.shape[0],
            QImage.Format_RGBA8888,
        )
        self.colorbarLabel.setPixmap(QPixmap.fromImage(image))

    def contrast_limits_slider_changed(self, slidermin, slidermax):
        valmin, valmax = self.layer._contrast_limits_range
        cmin = valmin + slidermin * (valmax - valmin)
        cmax = valmin + slidermax * (valmax - valmin)
        self.layer.contrast_limits = cmin, cmax

    def contrast_limits_slider_update(self):
        valmin, valmax = self.layer._contrast_limits_range
        cmin, cmax = self.layer.contrast_limits
        slidermin = (cmin - valmin) / (valmax - valmin)
        slidermax = (cmax - valmin) / (valmax - valmin)
        self.contrastLimitsSlider.blockSignals(True)
        self.contrastLimitsSlider.setValues((slidermin, slidermax))
        self.contrastLimitsSlider.blockSignals(False)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._contrast_limits_msg
