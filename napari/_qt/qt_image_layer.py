from qtpy.QtWidgets import QHBoxLayout
from . import QVRangeSlider
from .qt_base_layer import QtLayerControls, QtLayerProperties
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox
from ..layers.image._constants import Interpolation


class QtImageControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.clim.connect(lambda e: self.clim_slider_update())

        # Create clim slider
        self.climSlider = QVRangeSlider(
            slider_range=[0, 1, 0.0001], values=[0, 1], parent=self
        )
        self.climSlider.setEmitWhileMoving(True)
        self.climSlider.collapsable = False
        self.climSlider.setEnabled(True)

        layout = QHBoxLayout()
        layout.addWidget(self.climSlider)
        layout.setContentsMargins(12, 15, 10, 10)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.climSlider.rangeChanged.connect(self.clim_slider_changed)
        self.clim_slider_update()

    def clim_slider_changed(self, slidermin, slidermax):
        valmin, valmax = self.layer._clim_range
        cmin = valmin + slidermin * (valmax - valmin)
        cmax = valmin + slidermax * (valmax - valmin)
        self.layer.clim = cmin, cmax

    def clim_slider_update(self):
        valmin, valmax = self.layer._clim_range
        cmin, cmax = self.layer.clim
        slidermin = (cmin - valmin) / (valmax - valmin)
        slidermax = (cmax - valmin) / (valmax - valmin)
        self.climSlider.blockSignals(True)
        self.climSlider.setValues((slidermin, slidermax))
        self.climSlider.blockSignals(False)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._clim_msg


class QtImageProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.interpolation.connect(self._on_interpolation_change)

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

        row = self.grid_layout.rowCount()
        interp_comboBox = QComboBox()
        for interp in Interpolation:
            interp_comboBox.addItem(str(interp))
        index = interp_comboBox.findText(
            self.layer.interpolation, Qt.MatchFixedString
        )
        interp_comboBox.setCurrentIndex(index)
        interp_comboBox.activated[str].connect(
            lambda text=interp_comboBox: self.changeInterpolation(text)
        )
        self.interpComboBox = interp_comboBox
        self.grid_layout.addWidget(
            QLabel('interpolation:'), row, self.name_column
        )
        self.grid_layout.addWidget(interp_comboBox, row, self.property_column)

        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text

    def changeInterpolation(self, text):
        self.layer.interpolation = text

    def _on_interpolation_change(self, event):
        with self.layer.events.interpolation.blocker():
            index = self.interpComboBox.findText(
                self.layer.interpolation, Qt.MatchFixedString
            )
            self.interpComboBox.setCurrentIndex(index)

    def _on_colormap_change(self, event):
        name = self.layer.colormap[0]
        if name not in self.colormap_combobox._allitems:
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(name)
        if name != self.colormap_combobox.currentText():
            self.colormap_combobox.setCurrentText(name)
