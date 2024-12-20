from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets._image import (
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
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    attenuationLabel : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    attenuationSlider : qtpy.QtWidgets.QSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    autoScaleBar : qtpy.QtWidgets.QWidget
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    colormapWidget : qtpy.QtWidgets.QWidget
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    contrastLimitsSlider : superqt.QRangeSlider
        Contrast range slider widget.
    depictionComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current depiction value of the layer.
    depictionLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the depiction value chooser widget.
    gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    interpComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the interpolation mode for image display.
    interpLabel : qtpy.QtWidgets.QLabel
        Label for the interpolation dropdown menu.
    isoThresholdLabel : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
    isoThresholdSlider : qtpy.QtWidgets.QSlider
        Slider controlling the isosurface threshold value for rendering.
    layer : napari.layers.Image
        An instance of a napari Image layer.
    opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    planeNormalButtons : PlaneNormalButtons
        Buttons controlling plane normal orientation when the `plane` depiction value is choosed.
    planeNormalLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal value chooser widget.
    planeThicknessLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal thickness value chooser widget.
    planeThicknessSlider : superqt.QLabeledDoubleSlider
        Slider controlling plane normal thickness when the `plane` depiction value is choosed.
    renderComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the rendering mode for image display.
    renderLabel : qtpy.QtWidgets.QLabel
        Label for the rendering mode dropdown menu.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
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
            self._renderControl._on_display_change_hide()
            self._depictionControl._on_display_change_hide()
        else:
            self._renderControl._on_display_change_show()
            self._depictionControl._on_display_change_show()
        super()._on_ndisplay_changed()
