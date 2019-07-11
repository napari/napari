from qtpy.QtWidgets import QHBoxLayout
from . import QVRangeSlider
from .qt_base_layer import QtLayerControls, QtLayerProperties
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox


class QtVolumeControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.threshold.connect(self.threshold_slider_update)

        # Create threshold slider
        self.thresholdSlider = QVRangeSlider(
            slider_range=[0, 1, 0.0001], values=[0, 1], parent=self
        )
        self.thresholdSlider.setEmitWhileMoving(True)
        self.thresholdSlider.collapsable = False
        self.thresholdSlider.setEnabled(True)

        layout = QHBoxLayout()
        layout.addWidget(self.thresholdSlider)
        layout.setContentsMargins(12, 15, 10, 10)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.thresholdSlider.rangeChanged.connect(
            self.threshold_slider_changed
        )

    def threshold_slider_changed(self, slidermin, slidermax):
        valmin, valmax = self.layer._threshold_range
        cmin = valmin + slidermin * (valmax - valmin)
        cmax = valmin + slidermax * (valmax - valmin)
        self.layer.threshold = (cmin + cmax) / 4.444

    def threshold_slider_update(self, event):
        slidermin = 0
        slidermax = 1
        self.thresholdSlider.blockSignals(True)
        self.thresholdSlider.setValues((slidermin, slidermax))
        self.thresholdSlider.blockSignals(False)


class QtVolumeProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)

        row = self.grid_layout.rowCount()
        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        comboBox._allitems = set(self.layer.colormaps)
        index = comboBox.findText(self.layer.colormap[0], Qt.MatchFixedString)
        comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(
            lambda text=comboBox: self.changeColor(text)
        )
        self.grid_layout.addWidget(QLabel('colormap:'), row, self.name_column)
        self.grid_layout.addWidget(comboBox, row, self.property_column)
        self.colormap_combobox = comboBox

        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text

    def _on_colormap_change(self, event):
        name = self.layer.colormap[0]
        if name not in self.colormap_combobox._allitems:
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(name)
        if name != self.colormap_combobox.currentText():
            self.colormap_combobox.setCurrentText(name)
