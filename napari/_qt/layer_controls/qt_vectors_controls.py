from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtEdgeColorPropertyControl,
    QtLengthSpinBoxControl,
    QtOutSliceCheckBoxControl,
    QtVectorStyleComboBoxControl,
    QtWidthSpinBoxControl,
)
from napari.layers.base._base_constants import Mode

if TYPE_CHECKING:
    import napari.layers


class QtVectorsControls(QtLayerControls):
    """Qt view and controls for the napari Vectors layer.

    Parameters
    ----------
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.

    Attributes
    ----------
    layer : napari.layers.Vectors
        An instance of a napari Vectors layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button for pan/zoom mode.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.

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
    widthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spin box widget controlling edge line width of vectors.
    widthSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the edge line width of vectors chooser widget.
    lengthSpinBox : qtpy.QtWidgets.QDoubleSpinBox
        Spinbox widget controlling line length of vectors.
        Multiplicative factor on projections for length of all vectors.
    lengthSpinBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for line length of vectors chooser widget.
    vector_style_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select vector_style for the vectors.
    vector_style_comboBox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for vector_style value chooser widget.
    color_mode_comboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select edge_color_mode for the vectors.
    color_mode_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current selected edge_color_mode chooser widget.
    edgeColorEdit : qtpy.QtWidgets.QSlider
        Widget to select display color for vectors.
    edge_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge color chooser widget.
    color_prop_box : qtpy.QtWidgets.QComboBox
        Dropdown widget to select _edge_color_property for the vectors.
    edge_prop_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the _edge_color_property chooser widget.
    outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    outOfSliceCheckBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the out of slice display enablement chooser widget.
    """

    layer: 'napari.layers.Vectors'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._add_widget_controls(QtWidthSpinBoxControl(self, layer))
        self._add_widget_controls(QtLengthSpinBoxControl(self, layer))
        self._add_widget_controls(QtVectorStyleComboBoxControl(self, layer))
        self._add_widget_controls(QtEdgeColorPropertyControl(self, layer))
        self._add_widget_controls(QtOutSliceCheckBoxControl(self, layer))
