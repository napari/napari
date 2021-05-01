from contextlib import suppress
from functools import partial

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel, QPushButton, QSlider

from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.translations import trans
from ..utils import qt_signals_blocked
from ..widgets.qt_range_slider import QHRangeSlider
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
    contrastLimitsSlider : qtpy.QtWidgets.QHRangeSlider
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
        self.contrastLimitsSlider = QHRangeSlider(
            self.layer.contrast_limits,
            self.layer.contrast_limits_range,
            parent=self,
        )
        self.contrastLimitsSlider.mousePressEvent = self._clim_mousepress
        set_clim = partial(setattr, self.layer, 'contrast_limits')
        set_climrange = partial(setattr, self.layer, 'contrast_limits_range')
        self.contrastLimitsSlider.valuesChanged.connect(set_clim)
        self.contrastLimitsSlider.rangeChanged.connect(set_climrange)

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
            self.clim_pop = create_range_popup(
                self.layer, 'contrast_limits', parent=self
            )
            self.clim_pop.finished.connect(self.clim_pop.deleteLater)
            reset, fullrange = create_clim_reset_buttons(self.layer)
            self.clim_pop.layout.addWidget(reset)
            if fullrange is not None:
                self.clim_pop.layout.addWidget(fullrange)
            self.clim_pop.move_to('top', min_length=650)
            self.clim_pop.show()
        else:
            return QHRangeSlider.mousePressEvent(
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
            trans._(
                'Layer {layer} must have attribute {range_attr} to use "create_range_popup"',
                deferred=True,
                layer=layer,
                range_attr=range_attr,
            )
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
