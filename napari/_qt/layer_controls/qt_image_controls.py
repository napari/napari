from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets import (
    QtDepictionControl,
    QtImageRenderControl,
    QtInterpolationComboBoxControl,
)

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
    layer : napari.layers.Image
        An instance of a napari Image layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
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
    renderComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the rendering mode for image display.
    renderLabel : qtpy.QtWidgets.QLabel
        Label for the rendering mode dropdown menu.
    """

    layer: 'napari.layers.Image'
    PAN_ZOOM_ACTION_NAME = 'activate_image_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_image_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._interpolationControl = QtInterpolationComboBoxControl(
            self, layer
        )
        self._add_widget_controls(self._interpolationControl)
        self._depictionControl = QtDepictionControl(self, layer)
        self._add_widget_controls(self._depictionControl)
        self._renderControl = QtImageRenderControl(self, layer)
        self._add_widget_controls(self._renderControl)

        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        self._interpolationControl._update_interpolation_combo(self.ndisplay)
        self._depictionControl._update_plane_parameter_visibility()
        if self.ndisplay == 2:
            # self.isoThresholdSlider.hide()
            # self.isoThresholdLabel.hide()
            # self.attenuationSlider.hide()
            # self.attenuationLabel.hide()
            # self.renderComboBox.hide()
            # self.renderLabel.hide()
            # self.depictionComboBox.hide()
            # self.depictionLabel.hide()
            self._renderControl._on_display_change_hide()
            self._depictionControl._on_display_change_hide()
        else:
            # self.renderComboBox.show()
            # self.renderLabel.show()
            # self.depictionComboBox.show()
            # self.depictionLabel.show()
            self._renderControl._on_display_change_show()
            self._depictionControl._on_display_change_show()
        super()._on_ndisplay_changed()
