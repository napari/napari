from qtpy.QtWidgets import QHBoxLayout
from .. import QVRangeSlider
from .qt_base_layer import QtLayerControls
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox


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
        index = comboBox.findText(self.layer.colormap[0], Qt.MatchFixedString)
        comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(
            lambda text=comboBox: self.changeColor(text)
        )
        self.vbox_layout.addWidget(QLabel('colormap:'))
        self.vbox_layout.addWidget(comboBox)
        self.colormap_combobox = comboBox

        # Create contrast_limits slider
        self.contrastLimitsSlider = QVRangeSlider(
            slider_range=[0, 1, 0.0001], values=[0, 1], parent=self
        )
        self.contrastLimitsSlider.setEmitWhileMoving(True)
        self.contrastLimitsSlider.collapsable = False
        self.contrastLimitsSlider.setEnabled(True)

        self.vbox_layout.addWidget(self.contrastLimitsSlider)
        self.setMouseTracking(True)

        self.contrastLimitsSlider.rangeChanged.connect(
            self.contrast_limits_slider_changed
        )
        self.contrast_limits_slider_update()

    def changeColor(self, text):
        self.layer.colormap = text

    def _on_colormap_change(self, event):
        name = self.layer.colormap[0]
        if name not in self.colormap_combobox._allitems:
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(name)
        if name != self.colormap_combobox.currentText():
            self.colormap_combobox.setCurrentText(name)

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
