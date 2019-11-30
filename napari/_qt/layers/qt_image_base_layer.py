from qtpy.QtWidgets import QSlider
from .. import QHRangeSlider
from .qt_base_layer import QtLayerControls, QtLayerDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QCheckBox, QDoubleSpinBox
from qtpy.QtGui import QImage, QPixmap


class QtBaseImageControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.contrast_limits.connect(
            lambda e: self.contrast_limits_slider_update()
        )
        self.layer.events.gamma.connect(lambda e: self.gamma_slider_update())

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

        # gamma slider
        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(2)
        sld.setMaximum(200)
        sld.setSingleStep(2)
        sld.setValue(100)
        sld.valueChanged[int].connect(self.gamma_slider_changed)
        self.gammaSlider = sld
        self.gamma_slider_update()

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

    def gamma_slider_changed(self, value):
        self.layer.gamma = value / 100

    def gamma_slider_update(self):
        self.gammaSlider.blockSignals(True)
        self.gammaSlider.setValue(self.layer.gamma * 100)
        self.gammaSlider.blockSignals(False)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._contrast_limits_msg


class QtBaseImageDialog(QtLayerDialog):
    def __init__(self, layer):
        super().__init__(layer)

        self.colormapComboBox = QComboBox()
        for cmap in self.layer._colormaps:
            self.colormapComboBox.addItem(cmap)
        self.colormapComboBox._allitems = set(self.layer._colormaps)
        name = self.parameters['colormap'].default
        if name not in self.colormapComboBox._allitems:
            self.colormapComboBox._allitems.add(name)
            self.colormapComboBox.addItem(name)
        if name != self.colormapComboBox.currentText():
            self.colormapComboBox.setCurrentText(name)

        self.colormapCheckBox = QCheckBox(self)
        self.colormapCheckBox.setToolTip('Set colormap')
        self.colormapCheckBox.setChecked(False)
        self.colormapCheckBox.stateChanged.connect(self._on_colormap_change)
        self.colormapCheckBox.setChecked(False)
        self._on_colormap_change(None)

        self.gammaSpinBox = QDoubleSpinBox()
        self.gammaSpinBox.setToolTip('Gamma')
        self.gammaSpinBox.setKeyboardTracking(False)
        self.gammaSpinBox.setSingleStep(0.02)
        self.gammaSpinBox.setMinimum(0.02)
        self.gammaSpinBox.setMaximum(2)
        gamma = self.parameters['gamma'].default
        self.gammaSpinBox.setValue(gamma)

    def _on_colormap_change(self, event):
        state = self.colormapCheckBox.isChecked()
        if state:
            self.colormapComboBox.show()
        else:
            self.colormapComboBox.hide()
