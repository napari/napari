from functools import partial

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QComboBox, QLabel, QSlider, QPushButton

from ..qt_range_slider import QHRangeSlider
from ..qt_range_slider_popup import QRangeSliderPopup
from ..utils import qt_signals_blocked
from .qt_base_layer import QtLayerControls


class QtBaseImageControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.gamma.connect(self.gamma_slider_update)
        self.layer.events.contrast_limits.connect(self._on_clims_change)

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
            self.layer.contrast_limits, self.layer.contrast_limits_range
        )

        self.contrastLimitsSlider.mousePressEvent = self._clim_mousepress
        set_clim = partial(setattr, self.layer, 'contrast_limits')
        set_climrange = partial(setattr, self.layer, 'contrast_limits_range')
        self.contrastLimitsSlider.valuesChanged.connect(set_clim)
        self.contrastLimitsSlider.rangeChanged.connect(set_climrange)

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

    def _clim_mousepress(self, event):
        """when right clicking the clim slider, we want a QHRangeSliderPopup
        to appear that provides greater control over the values and range of
        the contrast limits.
        """
        if event.button() == Qt.RightButton:
            precision = 2
            try:
                layer_is_int = np.issubdtype(self.layer.data.dtype, np.integer)
                precision = 0 if layer_is_int else precision
            except AttributeError:
                layer_is_int = False

            self.clim_pop = QRangeSliderPopup(
                initial_values=self.layer.contrast_limits,
                data_range=self.layer.contrast_limits_range,
                collapsible=False,
                precision=precision,
                parent=self,
            )

            set_clim = partial(setattr, self.layer, 'contrast_limits')
            set_crange = partial(setattr, self.layer, 'contrast_limits_range')
            self.clim_pop.slider.valuesChanged.connect(set_clim)
            self.clim_pop.slider.rangeChanged.connect(set_crange)

            def reset():
                self.layer.reset_contrast_limits()
                self.layer.contrast_limits_range = self.layer.contrast_limits

            btn = QPushButton("reset")
            btn.setObjectName("reset_clims_button")
            btn.setToolTip("autoscale contrast to data range")
            btn.setFixedWidth(40)
            btn.clicked.connect(reset)
            self.clim_pop.layout.addWidget(btn)
            # the "full range" button doesn't do anything if it's not an
            # unsigned integer type (it's unclear what range should be set)
            # surface layers will also not have a dtype
            if np.issubdtype(self.layer.dtype, np.unsignedinteger):

                def reset_range():
                    self.layer.reset_contrast_limits_range()

                btn = QPushButton("full range")
                btn.setObjectName("full_clim_range_button")
                btn.setToolTip("set contrast range to full bit-depth")
                btn.setFixedWidth(65)
                btn.clicked.connect(reset_range)
                self.clim_pop.layout.addWidget(btn)
                self.clim_pop.finished.connect(self.clim_pop.deleteLater)

            self.clim_pop.show_at('top')
            self.clim_pop.curmin_label.update_position()
            self.clim_pop.curmax_label.update_position()
        else:
            return QHRangeSlider.mousePressEvent(
                self.contrastLimitsSlider, event
            )

    def _on_clims_change(self, event=None):
        with qt_signals_blocked(self.contrastLimitsSlider):
            self.contrastLimitsSlider.setRange(
                self.layer.contrast_limits_range
            )
            self.contrastLimitsSlider.setValues(self.layer.contrast_limits)

        try:
            self.clim_pop.slider.setRange(self.layer.contrast_limits_range)
            with qt_signals_blocked(self.clim_pop.slider):
                clims = self.layer.contrast_limits
                self.clim_pop.slider.setValues(clims)
                self.clim_pop._on_values_change(clims)
        # if popup has been deleted or doesn't exist
        except (AttributeError, RuntimeError):
            pass

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

    def gamma_slider_changed(self, value):
        self.layer.gamma = value / 100

    def gamma_slider_update(self, event=None):
        with qt_signals_blocked(self.gammaSlider):
            self.gammaSlider.setValue(self.layer.gamma * 100)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._contrast_limits_msg
