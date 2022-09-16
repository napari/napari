from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)
from superqt import QLabeledDoubleSlider

from ...layers.image._image_constants import (
    ImageRendering,
    Interpolation,
    VolumeDepiction,
)
from ...utils.action_manager import action_manager
from ...utils.translations import trans
from ..utils import qt_signals_blocked
from .qt_image_controls_base import QtBaseImageControls

if TYPE_CHECKING:
    import napari.layers


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

    layer: 'napari.layers.Image'

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.interpolation2d.connect(
            self._on_interpolation_change
        )
        self.layer.events.interpolation3d.connect(
            self._on_interpolation_change
        )
        self.layer.events.rendering.connect(self._on_rendering_change)
        self.layer.events.iso_threshold.connect(self._on_iso_threshold_change)
        self.layer.events.attenuation.connect(self._on_attenuation_change)
        self.layer.events._ndisplay.connect(self._on_ndisplay_change)
        self.layer.events.depiction.connect(self._on_depiction_change)
        self.layer.plane.events.thickness.connect(
            self._on_plane_thickness_change
        )

        self.interpComboBox = QComboBox(self)
        self.interpComboBox.currentTextChanged.connect(
            self.changeInterpolation
        )
        self.interpLabel = QLabel(trans._('interpolation:'))

        renderComboBox = QComboBox(self)
        rendering_options = [i.value for i in ImageRendering]
        renderComboBox.addItems(rendering_options)
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFlag.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.currentTextChanged.connect(self.changeRendering)
        self.renderComboBox = renderComboBox
        self.renderLabel = QLabel(trans._('rendering:'))

        self.depictionComboBox = QComboBox(self)
        depiction_options = [d.value for d in VolumeDepiction]
        self.depictionComboBox.addItems(depiction_options)
        index = self.depictionComboBox.findText(
            self.layer.depiction, Qt.MatchFlag.MatchFixedString
        )
        self.depictionComboBox.setCurrentIndex(index)
        self.depictionComboBox.currentTextChanged.connect(self.changeDepiction)
        self.depictionLabel = QLabel(trans._('depiction:'))

        # plane controls
        self.planeNormalButtons = PlaneNormalButtons(self)
        self.planeNormalLabel = QLabel(trans._('plane normal:'))
        action_manager.bind_button(
            'napari:orient_plane_normal_along_z',
            self.planeNormalButtons.zButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_y',
            self.planeNormalButtons.yButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_x',
            self.planeNormalButtons.xButton,
        )
        action_manager.bind_button(
            'napari:orient_plane_normal_along_view_direction',
            self.planeNormalButtons.obliqueButton,
        )

        self.planeThicknessSlider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal, self
        )
        self.planeThicknessLabel = QLabel(trans._('plane thickness:'))
        self.planeThicknessSlider.setFocusPolicy(Qt.NoFocus)
        self.planeThicknessSlider.setMinimum(1)
        self.planeThicknessSlider.setMaximum(50)
        self.planeThicknessSlider.setValue(self.layer.plane.thickness)
        self.planeThicknessSlider.valueChanged.connect(
            self.changePlaneThickness
        )

        sld = QSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(int(self.layer.iso_threshold * 100))
        sld.valueChanged.connect(self.changeIsoThreshold)
        self.isoThresholdSlider = sld
        self.isoThresholdLabel = QLabel(trans._('iso threshold:'))

        sld = QSlider(Qt.Orientation.Horizontal, parent=self)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(
            trans._('contrast limits:'), self.contrastLimitsSlider
        )
        self.layout().addRow(trans._('auto-contrast:'), self.autoScaleBar)
        self.layout().addRow(trans._('gamma:'), self.gammaSlider)
        self.layout().addRow(trans._('colormap:'), colormap_layout)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(self.interpLabel, self.interpComboBox)
        self.layout().addRow(self.depictionLabel, self.depictionComboBox)
        self.layout().addRow(self.renderLabel, self.renderComboBox)
        self.layout().addRow(self.isoThresholdLabel, self.isoThresholdSlider)
        self.layout().addRow(self.attenuationLabel, self.attenuationSlider)
        self.layout().addRow(self.planeNormalLabel, self.planeNormalButtons)
        self.layout().addRow(
            self.planeThicknessLabel, self.planeThicknessSlider
        )

    def changeInterpolation(self, text):
        """Change interpolation mode for image display.

        Parameters
        ----------
        text : str
            Interpolation mode used by vispy. Must be one of our supported
            modes:
            'bessel', 'bicubic', 'linear', 'blackman', 'catrom', 'gaussian',
            'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
            'nearest', 'spline16', 'spline36'
        """
        if self.layer._ndisplay == 2:
            self.layer.interpolation2d = text
        else:
            self.layer.interpolation3d = text

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

    def changeDepiction(self, text):
        self.layer.depiction = text
        self._toggle_plane_parameter_visibility()

    def changePlaneThickness(self, value: float):
        self.layer.plane.thickness = value

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

        with self.layer.events.interpolation.blocker(), self.layer.events.interpolation2d.blocker(), self.layer.events.interpolation3d.blocker():
            if self.interpComboBox.findText(interp_string) == -1:
                self.interpComboBox.addItem(interp_string)
            self.interpComboBox.setCurrentText(interp_string)

    def _on_rendering_change(self):
        """Receive layer model rendering change event and update dropdown menu."""
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFlag.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
            self._toggle_rendering_parameter_visbility()

    def _on_depiction_change(self):
        """Receive layer model depiction change event and update combobox."""
        with self.layer.events.depiction.blocker():
            index = self.depictionComboBox.findText(
                self.layer.depiction, Qt.MatchFlag.MatchFixedString
            )
            self.depictionComboBox.setCurrentIndex(index)
            self._toggle_plane_parameter_visibility()

    def _on_plane_thickness_change(self):
        with self.layer.plane.events.blocker():
            self.planeThicknessSlider.setValue(self.layer.plane.thickness)

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

    def _toggle_plane_parameter_visibility(self):
        """Hide plane rendering controls if they aren't needed."""
        depiction = VolumeDepiction(self.layer.depiction)
        if depiction == VolumeDepiction.VOLUME or self.layer._ndisplay == 2:
            self.planeNormalButtons.hide()
            self.planeNormalLabel.hide()
            self.planeThicknessSlider.hide()
            self.planeThicknessLabel.hide()
        if depiction == VolumeDepiction.PLANE and self.layer._ndisplay == 3:
            self.planeNormalButtons.show()
            self.planeNormalLabel.show()
            self.planeThicknessSlider.show()
            self.planeThicknessLabel.show()

    def _update_interpolation_combo(self):
        interp_names = [i.value for i in Interpolation.view_subset()]
        interp = (
            self.layer.interpolation2d
            if self.layer._ndisplay == 2
            else self.layer.interpolation3d
        )
        with qt_signals_blocked(self.interpComboBox):
            self.interpComboBox.clear()
            self.interpComboBox.addItems(interp_names)
            self.interpComboBox.setCurrentText(interp)

    def _on_ndisplay_change(self):
        """Toggle between 2D and 3D visualization modes."""
        self._update_interpolation_combo()
        self._toggle_plane_parameter_visibility()
        if self.layer._ndisplay == 2:
            self.isoThresholdSlider.hide()
            self.isoThresholdLabel.hide()
            self.attenuationSlider.hide()
            self.attenuationLabel.hide()
            self.renderComboBox.hide()
            self.renderLabel.hide()
            self.depictionComboBox.hide()
            self.depictionLabel.hide()
        else:
            self.renderComboBox.show()
            self.renderLabel.show()
            self._toggle_rendering_parameter_visbility()
            self.depictionComboBox.show()
            self.depictionLabel.show()


class PlaneNormalButtons(QWidget):
    """Qt buttons for controlling plane orientation.

        Attributes
    ----------
    xButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the x axis.
    yButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the y axis.
    zButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the z axis.
    obliqueButton : qtpy.QtWidgets.QPushButton
        Button which orients a plane normal along the camera view direction.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(2)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.xButton = QPushButton('x')
        self.yButton = QPushButton('y')
        self.zButton = QPushButton('z')
        self.obliqueButton = QPushButton(trans._('oblique'))

        self.layout().addWidget(self.xButton)
        self.layout().addWidget(self.yButton)
        self.layout().addWidget(self.zButton)
        self.layout().addWidget(self.obliqueButton)
