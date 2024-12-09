from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets import (
    QtImageRenderControl,
)
from napari._qt.layer_controls.widgets.image import (
    QtDepictionControl,
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

    Controls attributes
    -------------------
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    contrastLimitsSlider : superqt.QRangeSlider
        Contrast range slider widget.
    autoScaleBar : qtpy.QtWidgets.QWidget
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    colormapWidget : qtpy.QtWidgets.QWidget
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    interpComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the interpolation mode for image display.
    interpLabel : qtpy.QtWidgets.QLabel
        Label for the interpolation dropdown menu.
    depictionComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current depiction value of the layer.
    depictionLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the depiction value chooser widget.
    planeNormalButtons : PlaneNormalButtons
        Buttons controlling plane normal orientation when the `plane` depiction value is choosed.
    planeNormalLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal value chooser widget.
    planeThicknessSlider : superqt.QLabeledDoubleSlider
        Slider controlling plane normal thickness when the `plane` depiction value is choosed.
    planeThicknessLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the plane normal thickness value chooser widget.
    renderComboBox : qtpy.QtWidgets.QComboBox
        Dropdown menu to select the rendering mode for image display.
    renderLabel : qtpy.QtWidgets.QLabel
        Label for the rendering mode dropdown menu.
    attenuationSlider : qtpy.QtWidgets.QSlider
        Slider controlling attenuation rate for `attenuated_mip` mode.
    attenuationLabel : qtpy.QtWidgets.QLabel
        Label for the attenuation slider widget.
    isoThresholdSlider : qtpy.QtWidgets.QSlider
        Slider controlling the isosurface threshold value for rendering.
    isoThresholdLabel : qtpy.QtWidgets.QLabel
        Label for the isosurface threshold slider widget.
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
