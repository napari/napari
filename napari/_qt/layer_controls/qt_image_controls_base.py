from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from superqt import QDoubleRangeSlider

from napari._qt.layer_controls.qt_colormap_combobox import QtColormapComboBox
from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets._slider_compat import QDoubleSlider
from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup
from napari.utils._dtype import normalize_dtype
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.utils.events.event_utils import connect_no_arg, connect_setattr
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.layers import Image


class _QDoubleRangeSlider(QDoubleRangeSlider):
    def mousePressEvent(self, event):
        """Update the slider, or, on right-click, pop-up an expanded slider.

        The expanded slider provides finer control, directly editable values,
        and the ability to change the available range of the sliders.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.button() == Qt.MouseButton.RightButton:
            self.parent().show_clim_popupup()
        else:
            super().mousePressEvent(event)


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

    def __init__(self, layer: Image) -> None:
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.gamma.connect(self._on_gamma_change)
        self.layer.events.contrast_limits.connect(
            self._on_contrast_limits_change
        )
        self.layer.events.contrast_limits_range.connect(
            self._on_contrast_limits_range_change
        )

        comboBox = QtColormapComboBox(self)
        comboBox.setObjectName("colormapComboBox")
        comboBox._allitems = set(self.layer.colormaps)

        for name, cm in AVAILABLE_COLORMAPS.items():
            if name in self.layer.colormaps:
                comboBox.addItem(cm._display_name, name)

        comboBox.currentTextChanged.connect(self.changeColor)
        self.colormapComboBox = comboBox

        # Create contrast_limits slider
        self.contrastLimitsSlider = _QDoubleRangeSlider(
            Qt.Orientation.Horizontal, self
        )
        decimals = range_to_decimals(
            self.layer.contrast_limits_range, self.layer.dtype
        )
        self.contrastLimitsSlider.setRange(*self.layer.contrast_limits_range)
        self.contrastLimitsSlider.setSingleStep(10**-decimals)
        self.contrastLimitsSlider.setValue(self.layer.contrast_limits)
        self.contrastLimitsSlider.setToolTip(
            trans._('Right click for detailed slider popup.')
        )

        self.clim_popup = None

        connect_setattr(
            self.contrastLimitsSlider.valueChanged,
            self.layer,
            "contrast_limits",
        )
        connect_setattr(
            self.contrastLimitsSlider.rangeChanged,
            self.layer,
            'contrast_limits_range',
        )
        self.autoScaleBar = AutoScaleButtons(layer, self)

        # gamma slider
        sld = QDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self.layer.gamma)
        connect_setattr(sld.valueChanged, self.layer, 'gamma')
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
        with self.layer.events.colormap.blocker():
            self.layer.colormap = self.colormapComboBox.currentData()

    def _on_contrast_limits_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrastLimitsSlider):
            self.contrastLimitsSlider.setValue(self.layer.contrast_limits)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setValue(self.layer.contrast_limits)

    def _on_contrast_limits_range_change(self):
        """Receive layer model contrast limits change event and update slider."""
        with qt_signals_blocked(self.contrastLimitsSlider):
            decimals = range_to_decimals(
                self.layer.contrast_limits_range, self.layer.dtype
            )
            self.contrastLimitsSlider.setRange(
                *self.layer.contrast_limits_range
            )
            self.contrastLimitsSlider.setSingleStep(10**-decimals)

        if self.clim_popup:
            with qt_signals_blocked(self.clim_popup.slider):
                self.clim_popup.slider.setRange(
                    *self.layer.contrast_limits_range
                )

    def _on_colormap_change(self):
        """Receive layer model colormap change event and update dropdown menu."""
        name = self.layer.colormap.name
        if name not in self.colormapComboBox._allitems and (
            cm := AVAILABLE_COLORMAPS.get(name)
        ):
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

    def _on_gamma_change(self):
        """Receive the layer model gamma change event and update the slider."""
        with qt_signals_blocked(self.gammaSlider):
            self.gammaSlider.setValue(self.layer.gamma)

    def closeEvent(self, event):
        self.deleteLater()
        self.layer.events.disconnect(self)
        super().closeEvent(event)

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
        once_btn = QPushButton(trans._('once'))
        once_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        auto_btn = QPushButton(trans._('continuous'))
        auto_btn.setCheckable(True)
        auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        once_btn.clicked.connect(lambda: auto_btn.setChecked(False))
        connect_no_arg(once_btn.clicked, layer, "reset_contrast_limits")
        connect_setattr(auto_btn.toggled, layer, "_keep_auto_contrast")
        connect_no_arg(auto_btn.clicked, layer, "reset_contrast_limits")

        self.layout().addWidget(once_btn)
        self.layout().addWidget(auto_btn)

        # just for testing
        self._once_btn = once_btn
        self._auto_btn = auto_btn


class QContrastLimitsPopup(QRangeSliderPopup):
    def __init__(self, layer: Image, parent=None) -> None:
        super().__init__(parent)

        decimals = range_to_decimals(layer.contrast_limits_range, layer.dtype)
        self.slider.setRange(*layer.contrast_limits_range)
        self.slider.setDecimals(decimals)
        self.slider.setSingleStep(10**-decimals)
        self.slider.setValue(layer.contrast_limits)

        connect_setattr(self.slider.valueChanged, layer, "contrast_limits")
        connect_setattr(
            self.slider.rangeChanged, layer, "contrast_limits_range"
        )

        def reset():
            layer.reset_contrast_limits()
            layer.contrast_limits_range = layer.contrast_limits

        reset_btn = QPushButton("reset")
        reset_btn.setObjectName("reset_clims_button")
        reset_btn.setToolTip(trans._("autoscale contrast to data range"))
        reset_btn.setFixedWidth(45)
        reset_btn.clicked.connect(reset)
        self._layout.addWidget(
            reset_btn, alignment=Qt.AlignmentFlag.AlignBottom
        )

        # the "full range" button doesn't do anything if it's not an
        # unsigned integer type (it's unclear what range should be set)
        # so we don't show create it at all.
        if np.issubdtype(normalize_dtype(layer.dtype), np.integer):
            range_btn = QPushButton("full range")
            range_btn.setObjectName("full_clim_range_button")
            range_btn.setToolTip(
                trans._("set contrast range to full bit-depth")
            )
            range_btn.setFixedWidth(75)
            range_btn.clicked.connect(layer.reset_contrast_limits_range)
            self._layout.addWidget(
                range_btn, alignment=Qt.AlignmentFlag.AlignBottom
            )


def range_to_decimals(range_, dtype):
    """Convert a range to decimals of precision.

    Parameters
    ----------
    range_ : tuple
        Slider range, min and then max values.
    dtype : np.dtype
        Data type of the layer. Integers layers are given integer.
        step sizes.

    Returns
    -------
    int
        Decimals of precision.
    """

    if hasattr(dtype, 'numpy_dtype'):
        # retrieve the corresponding numpy.dtype from a tensorstore.dtype
        dtype = dtype.numpy_dtype

    if np.issubdtype(dtype, np.integer):
        return 0
    else:
        # scale precision with the log of the data range order of magnitude
        # eg.   0 - 1   (0 order of mag)  -> 3 decimal places
        #       0 - 10  (1 order of mag)  -> 2 decimals
        #       0 - 100 (2 orders of mag) -> 1 decimal
        #       ≥ 3 orders of mag -> no decimals
        # no more than 64 decimals
        d_range = np.subtract(*range_[::-1])
        return min(64, max(int(3 - np.log10(d_range)), 0))
