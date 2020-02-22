from functools import partial
from contextlib import suppress

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
        comboBox.addItems(self.layer.colormaps)
        comboBox._allitems = set(self.layer.colormaps)
        comboBox.activated[str].connect(self.changeColor)
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
        sld.valueChanged.connect(self.gamma_slider_changed)
        self.gammaSlider = sld
        self.gamma_slider_update()

        self.colorbarLabel = QLabel()
        self.colorbarLabel.setObjectName('colorbar')
        self.colorbarLabel.setToolTip('Colorbar')

        self._on_colormap_change()

    def changeColor(self, text):
        self.layer.colormap = text

    def _clim_mousepress(self, event):
        """Update the slider, or, on right-click, pop-up an expanded slider.

        The expanded slider provides finer control, directly editable values,
        and the ability to change the available range of the sliders.
        """
        if event.button() == Qt.RightButton:
            self.clim_pop = create_range_popup(
                self.layer, 'contrast_limits', self
            )
            self.clim_pop.finished.connect(self.clim_pop.deleteLater)
            reset, fullrange = create_clim_reset_buttons(self.layer)
            self.clim_pop.layout.addWidget(reset)
            if fullrange is not None:
                self.clim_pop.layout.addWidget(fullrange)
            self.clim_pop.show_at('top', min_length=650)
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

        # clim_popup will throw an AttributeError if not yet created
        # and a RuntimeError if it has already been cleaned up.
        # we only want to update the slider if it's active
        with suppress(AttributeError, RuntimeError):
            self.clim_pop.slider.setRange(self.layer.contrast_limits_range)
            with qt_signals_blocked(self.clim_pop.slider):
                clims = self.layer.contrast_limits
                self.clim_pop.slider.setValues(clims)
                self.clim_pop._on_values_change(clims)

    def _on_colormap_change(self, event=None):
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


def create_range_popup(layer, attr, parent=None):
    """Create a QRangeSliderPopup linked to a specific layer attribute.

    This assumes the layer has an attribute named both `attr` and `attr`_range.

    Parameters
    ----------
    layer : napari.layers.Layer
        probably an instance of Image or Surface layer
    attr : str
        the attribute to control with the slider.
    parent : QWidget
        probably an instance of QtLayerControls. important for styling.

    Returns
    -------
    QRangeSliderPopup

    Raises
    ------
    AttributeError
        if `layer` does not have an attribute named `{attr}_range`
    """
    range_attr = f'{attr}_range'
    if not hasattr(layer, range_attr):
        raise AttributeError(
            f'Layer {layer} must have attribute {range_attr} '
            'to use "create_range_popup"'
        )
    is_integer_type = np.issubdtype(layer.dtype, np.integer)

    d_range = getattr(layer, range_attr)
    popup = QRangeSliderPopup(
        initial_values=getattr(layer, attr),
        data_range=d_range,
        collapsible=False,
        precision=(
            0
            if is_integer_type
            # scale precision with the log of the data range order of magnitude
            # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
            #       0 - 10  (1 order of mag)  -> 2 decimals
            #       0 - 100 (2 orders of mag) -> 1 decimal
            #       ≥ 3 orders of mag -> no decimals
            else int(max(3 - np.log10(max(d_range[1] - d_range[0], 0.01)), 0))
        ),
        parent=parent,
    )

    set_values = partial(setattr, layer, attr)
    set_range = partial(setattr, layer, range_attr)
    popup.slider.valuesChanged.connect(set_values)
    popup.slider.rangeChanged.connect(set_range)
    return popup


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
    reset_btn.setToolTip("autoscale contrast to data range")
    reset_btn.setFixedWidth(40)
    reset_btn.clicked.connect(reset)

    range_btn = None
    # the "full range" button doesn't do anything if it's not an
    # unsigned integer type (it's unclear what range should be set)
    # so we don't show create it at all.
    if np.issubdtype(layer.dtype, np.integer):
        range_btn = QPushButton("full range")
        range_btn.setObjectName("full_clim_range_button")
        range_btn.setToolTip("set contrast range to full bit-depth")
        range_btn.setFixedWidth(65)
        range_btn.clicked.connect(reset_range)

    return reset_btn, range_btn
