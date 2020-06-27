from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider

from ...layers.image._image_constants import (
    Interpolation,
    Interpolation3D,
    Rendering,
)
from .qt_image_base_layer import QtBaseImageControls
from ...utils.event import Event


class QtImageControls(QtBaseImageControls):
    """Qt view and controls for the napari Image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    attenuationSlider : qtpy.QtWidgets.QSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    attenuationLabel : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    interpComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the interpolation mode for image display.
    interpLabel : qtpy.QtWidgets.QLabel
        Label for the interpolation dropdown menu.
    isoThresholdSlider : qtpy.QtWidgets.QSlider
        Slider controlling the isosurface threshold value for rendering.
    isoThresholdLabel : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
    layer : napari.layers.Image
        An instance of a napari Image layer.
    renderComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the rendering mode for image display.
    renderLabel : qtpy.QtWidgets.QLabel
        Label for the rendering mode dropdown menu.
    """

    def __init__(self, layer):
        super().__init__(layer)

        self.events.add(
            interpolation=Event,
            rendering=Event,
            iso_threshold=Event,
            attenuation=Event,
        )
        self.layer.dims.events.ndisplay.connect(
            lambda e: self._on_ndisplay_change(self.layer.dims.ndisplay)
        )

        self.interpComboBox = QComboBox(self)
        self.interpComboBox.activated[str].connect(self.events.interpolation)
        self.interpLabel = QLabel('interpolation:')

        renderComboBox = QComboBox(self)
        renderComboBox.addItems(Rendering.keys())
        renderComboBox.activated[str].connect(self.events.rendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel('rendering:')

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.valueChanged.connect(lambda v: self.events.iso_threshold(v / 100))
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QLabel('iso threshold:')

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(200)
        sld.setSingleStep(1)
        sld.valueChanged.connect(lambda v: self.events.attenuation(v / 100))
        self.attenuationSlider = sld
        self.attenuationLabel = QLabel('attenuation:')

        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(self.colorbarLabel)
        colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1)
        self.grid_layout.addWidget(QLabel('gamma:'), 2, 0)
        self.grid_layout.addWidget(self.gammaSlider, 2, 1)
        self.grid_layout.addWidget(QLabel('colormap:'), 3, 0)
        self.grid_layout.addLayout(colormap_layout, 3, 1)
        self.grid_layout.addWidget(QLabel('blending:'), 4, 0)
        self.grid_layout.addWidget(self.blendComboBox, 4, 1)
        self.grid_layout.addWidget(self.interpLabel, 5, 0)
        self.grid_layout.addWidget(self.interpComboBox, 5, 1)
        self.grid_layout.addWidget(self.renderLabel, 6, 0)
        self.grid_layout.addWidget(self.renderComboBox, 6, 1)
        self.grid_layout.addWidget(self.isoThresholdLabel, 7, 0)
        self.grid_layout.addWidget(self.isoThresholdSlider, 7, 1)
        self.grid_layout.addWidget(self.attenuationLabel, 8, 0)
        self.grid_layout.addWidget(self.attenuationSlider, 8, 1)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

        # Once EVH refactor is done, these can be moved to an initialization
        # outside of this object
        self._on_rendering_change(self.layer.rendering)
        self._on_iso_threshold_change(self.layer.iso_threshold)
        self._on_attenuation_change(self.layer.attenuation)
        self._on_ndisplay_change(self.layer.dims.ndisplay)

    def _on_interpolation_change(self, text):
        """Change interpolation mode for image display.

       Parameters
       ----------
       text : str
           Interpolation mode used by VisPy. Must be one of our supported
           modes:
           'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
           'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
           'nearest', 'spline16', 'spline36'
       """
        index = self.interpComboBox.findText(text, Qt.MatchFixedString)
        self.interpComboBox.setCurrentIndex(index)

    def _on_iso_threshold_change(self, value):
        """Receive layer model isosurface change event and update the slider.

        Parameters
        ----------
        value : float
            Iso surface threshold value, between 0 and 1.
        """
        self.isoThresholdSlider.setValue(value * 100)

    def _on_attenuation_change(self, value):
        """Receive layer model attenuation change event and update the slider.

        Parameters
        ----------
        value : float
            Attenuation value, between 0 and 2.
        """
        self.attenuationSlider.setValue(value * 100)

    def _on_rendering_change(self, text):
        """Receive layer model rendering change event and update dropdown menu.

        Parameters
        ----------
        text : str
            Rendering mode used by VisPy.
            Selects a preset rendering mode in VisPy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maxiumum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        index = self.renderComboBox.findText(text, Qt.MatchFixedString)
        self.renderComboBox.setCurrentIndex(index)
        self._toggle_rendering_parameter_visbility()

    def _toggle_rendering_parameter_visbility(self):
        """Hide isosurface rendering parameters if they aren't needed."""
        text = self.renderComboBox.currentText()
        rendering = Rendering(text)
        if rendering == Rendering.ISO:
            self.isoThresholdSlider.show()
            self.isoThresholdLabel.show()
        else:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
        if rendering == Rendering.ATTENUATED_MIP:
            self.attenuationSlider.show()
            self.attenuationLabel.show()
        else:
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()

    def _update_interpolation_combo(self, ndisplay):
        """Set allowed interploation modes for dimensionality of display.

        Parameters
        ----------
        ndisplay : int
            Number of dimesnions to be displayed, must be `2` or `3`.
        """
        interp_enum = Interpolation if ndisplay == 2 else Interpolation3D
        self.interpComboBox.clear()
        self.interpComboBox.addItems(interp_enum.keys())
        # To finish EVH refactor we need to revisit the coupling of 2D and
        # 3D interpolation modes into one attribute
        self._on_interpolation_change(self.layer.interpolation)

    def _on_ndisplay_change(self, value):
        """Toggle between 2D and 3D visualization modes.

        Parameters
        ----------
        value : int
            Number of dimesnions to be displayed, must be `2` or `3`.
        """
        self._update_interpolation_combo(value)
        if value == 2:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()
            self.renderComboBox.hide()
            self.renderLabel.hide()
        else:
            self.renderComboBox.show()
            self.renderLabel.show()
            self._toggle_rendering_parameter_visbility()
