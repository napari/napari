from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider

from ...layers.image._image_constants import (
    ImageRendering,
    Interpolation,
    Interpolation3D,
)
from ...utils.translations import trans
from .qt_image_controls_base import QtBaseImageControls


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

        self.layer.events.interpolation.connect(self._on_interpolation_change)
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        self.layer.events._ndisplay.connect(self._on_ndisplay_change)

        self.interpComboBox = QComboBox(self)
        self.interpComboBox.activated[str].connect(self.changeInterpolation)
        self.interpLabel = QLabel(trans._('interpolation:'))

        renderComboBox = QComboBox(self)
        rendering_options = [i.value for i in ImageRendering]
        renderComboBox.addItems(rendering_options)
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.activated[str].connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel(trans._('rendering:'))

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(int(self.layer.iso_threshold * 100))
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QLabel(trans._('iso threshold:'))

        sld = QSlider(Qt.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(int(self.layer.attenuation * 200))
        sld.valueChanged.connect(self.changeAttenuation)
        self.attenuationSlider = sld
        self.attenuationLabel = QLabel(trans._('attenuation:'))
        self._on_ndisplay_change()

        colormap_layout = QHBoxLayout()
        if hasattr(self.layer, 'rgb') and self.layer.rgb:
            colormap_layout.addWidget(QLabel("RGB"))
            self.colormapComboBox.setVisible(False)
            self.colorbarLabel.setVisible(False)
        else:
            colormap_layout.addWidget(self.colorbarLabel)
            colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel(trans._('contrast limits:')), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('auto-contrast:')), 2, 0)
        self.grid_layout.addWidget(self.autoScaleBar, 2, 1)
        self.grid_layout.addWidget(QLabel(trans._('gamma:')), 3, 0)
        self.grid_layout.addWidget(self.gammaSlider, 3, 1)
        self.grid_layout.addWidget(QLabel(trans._('colormap:')), 4, 0)
        self.grid_layout.addLayout(colormap_layout, 4, 1)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 5, 0)
        self.grid_layout.addWidget(self.blendComboBox, 5, 1)
        self.grid_layout.addWidget(self.interpLabel, 6, 0)
        self.grid_layout.addWidget(self.interpComboBox, 6, 1)
        self.grid_layout.addWidget(self.renderLabel, 7, 0)
        self.grid_layout.addWidget(self.renderComboBox, 7, 1)
        self.grid_layout.addWidget(self.isoThresholdLabel, 8, 0)
        self.grid_layout.addWidget(self.isoThresholdSlider, 8, 1)
        self.grid_layout.addWidget(self.attenuationLabel, 9, 0)
        self.grid_layout.addWidget(self.attenuationSlider, 9, 1)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def changeInterpolation(self, text):
        """Change interpolation mode for image display.

        Parameters
        ----------
        text : str
            Interpolation mode used by vispy. Must be one of our supported
            modes:
            'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
        """
        self.layer.interpolation = text

    def changeRendering(self, text):
        """Change rendering mode for image display.

        Parameters
        ----------
        text : str
            Rendering mode used by vispy.
            Selects a preset rendering mode in vispy that determines how
            volume is displayed:
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
        """
        self.layer.rendering = text
        self._toggle_rendering_parameter_visbility()

    def changeIsoThreshold(self, value):
        """Change isosurface threshold on the layer model.

        Parameters
        ----------
        value : float
            Threshold for isosurface.
        """
        with self.layer.events.blocker(self._on_iso_threshold_change):
            self.layer.iso_threshold = value / 100

    def _on_iso_threshold_change(self):
        """Receive layer model isosurface change event and update the slider."""
        with self.layer.events.iso_threshold.blocker():
            self.isoThresholdSlider.setValue(
                int(self.layer.iso_threshold * 100)
            )

    def changeAttenuation(self, value):
        """Change attenuation rate for attenuated maximum intensity projection.

        Parameters
        ----------
        value : Float
            Attenuation rate for attenuated maximum intensity projection.
        """
        with self.layer.events.blocker(self._on_attenuation_change):
            self.layer.attenuation = value / 200

    def _on_attenuation_change(self):
        """Receive layer model attenuation change event and update the slider."""
        with self.layer.events.attenuation.blocker():
            self.attenuationSlider.setValue(int(self.layer.attenuation * 200))

    def _on_interpolation_change(self, event):
        """Receive layer interpolation change event and update dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        interp_string = event.value.value

        with self.layer.events.interpolation.blocker():
            if self.interpComboBox.findText(interp_string) == -1:
                self.interpComboBox.addItem(interp_string)
            self.interpComboBox.setCurrentText(interp_string)

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
            self._toggle_rendering_parameter_visbility()

    def _toggle_rendering_parameter_visbility(self):
        """Hide isosurface rendering parameters if they aren't needed."""
        rendering = ImageRendering(self.layer.rendering)
        if rendering == ImageRendering.ISO:
            self.isoThresholdSlider.show()
            self.isoThresholdLabel.show()
        else:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
        if rendering == ImageRendering.ATTENUATED_MIP:
            self.attenuationSlider.show()
            self.attenuationLabel.show()
        else:
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()

    def _update_interpolation_combo(self):
        self.interpComboBox.clear()
        interp_names = (
            Interpolation3D.keys()
            if self.layer._ndisplay == 3
            else [i.value for i in Interpolation.view_subset()]
        )
        self.interpComboBox.addItems(interp_names)
        index = self.interpComboBox.findText(
            self.layer.interpolation, Qt.MatchFixedString
        )
        self.interpComboBox.setCurrentIndex(index)

    def _on_ndisplay_change(self):
        """Toggle between 2D and 3D visualization modes."""
        self._update_interpolation_combo()
        if self.layer._ndisplay == 2:
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
