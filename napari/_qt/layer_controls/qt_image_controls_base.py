from functools import partial

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel, QPushButton, QSlider
from qtrangeslider import QRangeSlider

from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.translations import trans
from ..utils import qt_signals_blocked
from ..widgets.qt_range_slider_popup import QRangeSliderPopup
from .qt_colormap_combobox import QtColormapComboBox
from .qt_layer_controls_base import QtLayerControls


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
    clim_pop : napari._qt.qt_range_slider_popup.QRangeSliderPopup
        Popup widget launching the contrast range slider.
    colorbarLabel : qtpy.QtWidgets.QLabel
        Label text of colorbar widget.
    colormapComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget for selecting the layer colormap.
    contrastLimitsSlider : qtrangeslider.QRangeSlider
        Contrast range slider widget.
    gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    layer : napari.layers.Layer
        An instance of a napari layer.

    """

    def __init__(self, layer):
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
        self.contrastLimitsSlider = QRangeSlider(Qt.Horizontal, self)
        self.contrastLimitsSlider.setRange(*self.layer.contrast_limits_range)
        self.contrastLimitsSlider.setValue(self.layer.contrast_limits)

        self.contrastLimitsSlider.mousePressEvent = self._clim_mousepress
        set_clim = partial(setattr, self.layer, 'contrast_limits')

        self.contrastLimitsSlider.valueChanged.connect(set_clim)
        self.contrastLimitsSlider.rangeChanged.connect(
            lambda *a: setattr(self.layer, 'contrast_limits_range', a)
        )

        # gamma slider
        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(2)
        sld.setMaximum(200)
        sld.setSingleStep(2)
        sld.setValue(100)
        sld.valueChanged.connect(self.gamma_slider_changed)
        self.gammaSlider = sld
        self._on_gamma_change()

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
            self.open_clim_popup()
        else:
            QRangeSlider.mousePressEvent(self.contrastLimitsSlider, event)

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

    def gamma_slider_changed(self, value):
        """Change gamma value on the layer model.

        Parameters
        ----------
        value : float
            Gamma adjustment value.
            https://en.wikipedia.org/wiki/Gamma_correction
        """
        self.layer.gamma = value / 100

    def _on_gamma_change(self, event=None):
        """Receive the layer model gamma change event and update the slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.gammaSlider):
            self.gammaSlider.setValue(int(self.layer.gamma * 100))

    def closeEvent(self, event):
        self.deleteLater()
        event.accept()

    def open_clim_popup(self):
        clim_pop = QRangeSliderPopup(self)
        clim_pop.slider.setRange(*self.layer.contrast_limits_range)
        clim_pop.slider.setValue(self.layer.contrast_limits)

        set_values = partial(setattr, self.layer, 'contrast_limits')
        clim_pop.slider.valueChanged.connect(set_values)

        def set_range(min_, max_):
            self.layer.contrast_limits_range = (min_, max_)

        clim_pop.slider.rangeChanged.connect(set_range)

        reset, fullrange = create_clim_reset_buttons(self.layer)
        clim_pop._layout.addWidget(reset)
        if fullrange is not None:
            clim_pop._layout.addWidget(fullrange)
        clim_pop.move_to('top', min_length=650)
        clim_pop.open()


def create_clim_reset_buttons(layer):
    """Create contrast limits reset and full range buttons.

    Important: consumers of this function should check whether range_btn is
    not None before adding the widget to a layout.  Adding None to a layout
    can cause a segmentation fault.

    Parameters
    ----------
    layer : napari.layers.Layer
        Image or Surface Layer

    Returns
    -------
    2-tuple
        If layer data type is integer type, returns (reset_btn, range_btn).
        Else, returns (reset_btn, None)
    """

    def reset():
        layer.reset_contrast_limits()
        layer.contrast_limits_range = layer.contrast_limits

    def reset_range():
        layer.reset_contrast_limits_range()

    reset_btn = QPushButton("reset")
    reset_btn.setObjectName("reset_clims_button")
    reset_btn.setToolTip(trans._("autoscale contrast to data range"))
    reset_btn.setFixedWidth(40)
    reset_btn.clicked.connect(reset)

    range_btn = None
    # the "full range" button doesn't do anything if it's not an
    # unsigned integer type (it's unclear what range should be set)
    # so we don't show create it at all.
    if np.issubdtype(layer.dtype, np.integer):
        range_btn = QPushButton("full range")
        range_btn.setObjectName("full_clim_range_button")
        range_btn.setToolTip(trans._("set contrast range to full bit-depth"))
        range_btn.setFixedWidth(65)
        range_btn.clicked.connect(reset_range)

    return reset_btn, range_btn
