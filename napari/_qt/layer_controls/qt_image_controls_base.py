from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from superqt import QDoubleRangeSlider
from superqt import QLabeledDoubleSlider as QSlider

from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.translations import trans
from ..utils import qt_signals_blocked
from ..widgets.qt_range_slider_popup import QRangeSliderPopup
from .qt_colormap_combobox import QtColormapComboBox
from .qt_layer_controls_base import QtLayerControls

if TYPE_CHECKING:
    from napari.layers import Image


class QtBaseImageControls(QtLayerControls):
    """Superclass for classes requiring colormaps, contrast & gamma sliders.

    This class is never directly instantiated anywhere.
    It is subclassed by QtImageControls and QtSurfaceControls.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    clim_popup : napari._qt.qt_range_slider_popup.QRangeSliderPopup
        Popup widget launching the contrast range slider.
    colorbarLabel : qtpy.QtWidgets.QLabel
        Label text of colorbar widget.
    colormapComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget for selecting the layer colormap.
    contrastLimitsSlider : superqt.QRangeSlider
        Contrast range slider widget.
    gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    layer : napari.layers.Layer
        An instance of a napari layer.

    """

    def __init__(self, layer: Image):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )

        comboBox = QtColormapComboBox(self)
        comboBox.setObjectName("colormapComboBox")
        comboBox._allitems = set(self.layer.colormaps)

        for name, cm in AVAILABLE_COLORMAPS.items():
            if name in self.layer.colormaps:
                comboBox.addItem(cm._display_name, name)

        comboBox.activated[str].connect(self.changeColor)
        self.colormapComboBox = comboBox

        # Create contrast_limits slider
        self.contrastLimitsSlider = QDoubleRangeSlider(Qt.Horizontal, self)
        self.contrastLimitsSlider.setSingleStep(0.01)
        self.contrastLimitsSlider.setRange(*self.layer.contrast_limits_range)
        self.contrastLimitsSlider.setValue(self.layer.contrast_limits)
        self.contrastLimitsSlider.setToolTip(
            trans._('Right click for detailed slider popup.')
        )

        self.clim_popup = None

        self.contrastLimitsSlider.mousePressEvent = self._clim_mousepress
        set_clim = partial(setattr, self.layer, 'contrast_limits')

        self.contrastLimitsSlider.valueChanged.connect(set_clim)
        self.contrastLimitsSlider.rangeChanged.connect(
            lambda *a: setattr(self.layer, 'contrast_limits_range', a)
        )
        self.autoScaleBar = AutoScaleButtons(layer, self)

        # gamma slider
        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self.layer.gamma)
        sld.valueChanged.connect(lambda v: setattr(self.layer, 'gamma', v))
        self.gammaSlider = sld

        self.colorbarLabel = QLabel(parent=self)
        self.colorbarLabel.setObjectName('colorbar')
        self.colorbarLabel.setToolTip(trans._('Colorbar'))

        self._on_colormap_change()

    def changeColor(self, text):
        """Change colormap on the layer model.

        Parameters
        ----------
        text : str
            Colormap name.
        """
        self.layer.colormap = self.colormapComboBox.currentData()

    def _clim_mousepress(self, event):
        """Update the slider, or, on right-click, pop-up an expanded slider.

        The expanded slider provides finer control, directly editable values,
        and the ability to change the available range of the sliders.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.button() == Qt.RightButton:
            self.show_clim_popupup()
        else:
            QDoubleRangeSlider.mousePressEvent(
                self.contrastLimitsSlider, event
            )

    def _on_contrast_limits_change(self, event=None):
        """Receive layer model contrast limits change event and update slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.contrastLimitsSlider):
            self.contrastLimitsSlider.setRange(
                *self.layer.contrast_limits_range
            )
            self.contrastLimitsSlider.setValue(self.layer.contrast_limits)

        if self.clim_popup:
            self.clim_popup.slider.setRange(*self.layer.contrast_limits_range)
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(self.layer.contrast_limits)

    def _on_colormap_change(self, event=None):
        """Receive layer model colormap change event and update dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        name = self.layer.colormap.name
        if name not in self.colormapComboBox._allitems:
            cm = AVAILABLE_COLORMAPS.get(name)
            if cm:
                self.colormapComboBox._allitems.add(name)
                self.colormapComboBox.addItem(cm._display_name, name)

        if name != self.colormapComboBox.currentData():
            index = self.colormapComboBox.findData(name)
            self.colormapComboBox.setCurrentIndex(index)

        # Note that QImage expects the image width followed by height
        cbar = self.layer.colormap.colorbar
        image = QImage(
            cbar,
            cbar.shape[1],
            cbar.shape[0],
            QImage.Format_RGBA8888,
        )
        self.colorbarLabel.setPixmap(QPixmap.fromImage(image))

    def _on_gamma_change(self, event=None):
        """Receive the layer model gamma change event and update the slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.gammaSlider):
            self.gammaSlider.setValue(self.layer.gamma)

    def closeEvent(self, event):
        self.deleteLater()
        event.accept()

    def show_clim_popupup(self):
        self.clim_popup = QContrastLimitsPopup(self.layer, self)
        self.clim_popup.setParent(self)
        self.clim_popup.move_to('top', min_length=650)
        self.clim_popup.show()


class AutoScaleButtons(QWidget):
    def __init__(self, layer: Image, parent=None) -> None:
        super().__init__(parent=parent)

        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)
        once_btn = QPushButton('once')
        once_btn.setFocusPolicy(Qt.NoFocus)

        auto_btn = QPushButton('continuous')
        auto_btn.setCheckable(True)
        auto_btn.setFocusPolicy(Qt.NoFocus)
        once_btn.clicked.connect(lambda: auto_btn.setChecked(False))
        once_btn.clicked.connect(lambda: layer.reset_contrast_limits())
        auto_btn.toggled.connect(
            lambda e: setattr(layer, '_keep_autoscale', e)
        )
        auto_btn.clicked.connect(
            lambda e: layer.reset_contrast_limits() if e else None
        )

        self.layout().addWidget(once_btn)
        self.layout().addWidget(auto_btn)


class QContrastLimitsPopup(QRangeSliderPopup):
    def __init__(self, layer: Image, parent=None):
        super().__init__(parent)

        if np.issubdtype(layer.dtype, np.integer):
            decimals = 0
        else:
            # scale precision with the log of the data range order of magnitude
            # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
            #       0 - 10  (1 order of mag)  -> 2 decimals
            #       0 - 100 (2 orders of mag) -> 1 decimal
            #       ≥ 3 orders of mag -> no decimals
            # no more than 6 decimals
            d_range = np.subtract(*layer.contrast_limits_range[::-1])
            decimals = min(6, max(int(3 - np.log10(d_range)), 0))

        self.slider.setRange(*layer.contrast_limits_range)
        self.slider.setDecimals(decimals)
        self.slider.setSingleStep(10 ** -decimals)
        self.slider.setValue(layer.contrast_limits)

        set_values = partial(setattr, layer, 'contrast_limits')
        self.slider.valueChanged.connect(set_values)
        self.slider.rangeChanged.connect(
            lambda *a: setattr(layer, 'contrast_limits_range', a)
        )

        def reset():
            layer.reset_contrast_limits()
            layer.contrast_limits_range = layer.contrast_limits

        reset_btn = QPushButton("reset")
        reset_btn.setObjectName("reset_clims_button")
        reset_btn.setToolTip(trans._("autoscale contrast to data range"))
        reset_btn.setFixedWidth(40)
        reset_btn.clicked.connect(reset)
        self._layout.addWidget(reset_btn, alignment=Qt.AlignBottom)

        # the "full range" button doesn't do anything if it's not an
        # unsigned integer type (it's unclear what range should be set)
        # so we don't show create it at all.
        if np.issubdtype(layer.dtype, np.integer):
            range_btn = QPushButton("full range")
            range_btn.setObjectName("full_clim_range_button")
            range_btn.setToolTip(
                trans._("set contrast range to full bit-depth")
            )
            range_btn.setFixedWidth(65)
            range_btn.clicked.connect(layer.reset_contrast_limits_range)
            self._layout.addWidget(range_btn, alignment=Qt.AlignBottom)
