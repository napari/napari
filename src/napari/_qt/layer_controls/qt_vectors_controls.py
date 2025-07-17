from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtOutSliceCheckBoxControl,
)
from napari._qt.layer_controls.widgets._vectors import (
    QtEdgeColorPropertyControl,
    QtLengthSpinBoxControl,
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
    _edge_color_property_control : napari._qt.layer_controls.widgets._vectors.QtEdgeColorPropertyControl
        Widget that wraps the widgets used to select vectors edge color mode, property and color.
    _length_spinbox_control : napari._qt.layer_controls.widgets._vectors.QtLengthSpinBoxControl
        Widget that wraps a spinbox widget controlling length of vectors.
    _out_slice_checkbox_control : napari._qt.layer_controls.widgets.QtOutSliceCheckBoxControl
        Widget that wraps a checkbox to indicate whether to render out of slice.
    _vector_style_combobox_control : napari._qt.layer_controls.widgets._vectors.QtVectorStyleComboBoxControl
        Widget that wraps a dropdown widget to select vector_style for the vectors.
    _width_spinbox_control : napari._qt.layer_controls.widgets._vectors.QtWidthSpinBoxControl
        Widget that wraps a spinbox controlling edge line width of vectors.
    """

    layer: 'napari.layers.Vectors'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._width_spinbox_control = QtWidthSpinBoxControl(self, layer)
        self._add_widget_controls(self._width_spinbox_control)
        self._length_spinbox_control = QtLengthSpinBoxControl(self, layer)
        self._add_widget_controls(self._length_spinbox_control)
        self._vector_style_combobox_control = QtVectorStyleComboBoxControl(
            self, layer
        )
        self._add_widget_controls(self._vector_style_combobox_control)
        self._edge_color_property_control = QtEdgeColorPropertyControl(
            self, layer
        )
        self._add_widget_controls(self._edge_color_property_control)
        self._out_slice_checkbox_control = QtOutSliceCheckBoxControl(
            self, layer
        )
        self._add_widget_controls(self._out_slice_checkbox_control)
