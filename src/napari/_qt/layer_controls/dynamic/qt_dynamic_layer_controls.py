from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from napari._qt.layer_controls.dynamic.buttons.qt_image_buttons import (
    QtImageButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_labels_buttons import (
    QtLabelsButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_layer_buttons_base import (
    QtLayerButtons,
    QtMultiLayerButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_points_buttons import (
    QtPointsButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_shapes_buttons import (
    QtShapesButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_surface_buttons import (
    QtSurfaceButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_tracks_buttons import (
    QtTracksButtons,
)
from napari._qt.layer_controls.dynamic.buttons.qt_vectors_buttons import (
    QtVectorsButtons,
)
from napari._qt.layer_controls.dynamic.widgets import (
    QtOpacityBlendingControls,
    QtWidgetControlsBase,
)
from napari._qt.layer_controls.dynamic.widgets._image.qt_depiction_control import (
    QtDepictionControl,
)
from napari._qt.layer_controls.dynamic.widgets._image.qt_interpolation_combobox import (
    QtInterpolationComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._image.qt_render_control import (
    QtImageRenderControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_brush_size_slider import (
    QtBrushSizeSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_color_mode_combobox import (
    QtColorModeComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_contiguous_checkbox import (
    QtContiguousCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_contour_spinbox import (
    QtContourSpinBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_current_label_controls import (
    QtCurrentLabelControls,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_display_selected_label_checkbox import (
    QtDisplaySelectedLabelCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_ndim_spinbox import (
    QtNdimSpinBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_preserve_labels_checkbox import (
    QtPreserveLabelsCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._labels.qt_rendering_control import (
    QtLabelRenderingControl,
)
from napari._qt.layer_controls.dynamic.widgets._points.qt_border_color import (
    QtBorderColorControl,
)
from napari._qt.layer_controls.dynamic.widgets._points.qt_current_size_slider import (
    QtCurrentSizeSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets._points.qt_symbol_combobox import (
    QtSymbolComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._shapes.qt_edge_color import (
    QtEdgeColorControl,
)
from napari._qt.layer_controls.dynamic.widgets._shapes.qt_edge_width_slider import (
    QtEdgeWidthSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets._surface.qt_shading_combobox import (
    QtShadingComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_color_properties_combobox import (
    QtColorPropertiesComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_colormap_control import (
    QtColormapComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_graph_checkbox import (
    QtGraphCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_head_slider import (
    QtHeadLengthSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_hide_completed_tracks_checkbox import (
    QtHideCompletedTracksCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_id_checkbox import (
    QtIdCheckBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._tracks.qt_tail_control import (
    QtTailDisplayCheckBoxControl,
    QtTailLengthSliderControl,
    QtTailWidthSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets._vectors.qt_edge_color import (
    QtEdgeColorFeatureControl,
)
from napari._qt.layer_controls.dynamic.widgets._vectors.qt_line_dimension_spinbox import (
    QtLengthSpinBoxControl,
    QtWidthSpinBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._vectors.qt_vector_style_combobox import (
    QtVectorStyleComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_colormap_control import (
    QtColormapControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_contrast_limits import (
    QtContrastLimitsControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_face_color import (
    QtFaceColorControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_gamma_slider import (
    QtGammaSliderControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_histogram_control import (
    QtHistogramControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_multiscale_level_control import (
    QtMultiscaleLevelControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_projection_mode_control import (
    QtProjectionModeControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_text_visibility import (
    QtTextVisibilityControl,
)
from napari._qt.utils import set_widgets_enabled_with_opacity
from napari.layers import (
    Image,
    Labels,
    Layer,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers.base._base_constants import Mode
from napari.layers.intensity_mixin import IntensityVisualizationMixin
from napari.utils.events import disconnect_events

controls_dict = {
    Layer: (
        QtOpacityBlendingControls,
        QtMultiscaleLevelControl,
        QtProjectionModeControl,
    ),
    IntensityVisualizationMixin: (
        QtContrastLimitsControl,
        QtHistogramControl,
        QtGammaSliderControl,
        QtColormapControl,
    ),
    Points | Shapes: (
        QtTextVisibilityControl,
        QtFaceColorControl,
    ),
    Points: (
        QtBorderColorControl,
        QtCurrentSizeSliderControl,
        QtSymbolComboBoxControl,
    ),
    Surface: (QtShadingComboBoxControl,),
    Labels: (
        QtBrushSizeSliderControl,
        QtColorModeComboBoxControl,
        QtContiguousCheckBoxControl,
        QtContourSpinBoxControl,
        QtDisplaySelectedLabelCheckBoxControl,
        QtCurrentLabelControls,
        QtNdimSpinBoxControl,
        QtPreserveLabelsCheckBoxControl,
        QtLabelRenderingControl,
    ),
    Image: (
        QtDepictionControl,
        QtInterpolationComboBoxControl,
        QtImageRenderControl,
    ),
    Shapes: (
        QtEdgeColorControl,
        QtEdgeWidthSliderControl,
    ),
    Tracks: (
        QtColorPropertiesComboBoxControl,
        QtColormapComboBoxControl,
        QtGraphCheckBoxControl,
        QtHeadLengthSliderControl,
        QtHideCompletedTracksCheckBoxControl,
        QtIdCheckBoxControl,
        QtTailLengthSliderControl,
        QtTailWidthSliderControl,
        QtTailDisplayCheckBoxControl,
    ),
    Vectors: (
        QtEdgeColorFeatureControl,
        QtWidthSpinBoxControl,
        QtLengthSpinBoxControl,
        QtVectorStyleComboBoxControl,
    ),
}

buttons_dict = {
    Image: QtImageButtons,
    Surface: QtSurfaceButtons,
    Labels: QtLabelsButtons,
    Points: QtPointsButtons,
    Shapes: QtShapesButtons,
    Vectors: QtVectorsButtons,
    Tracks: QtTracksButtons,
}


class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )


EXPERIMENTAL_WARNING = """Dynamically generated layer controls are a new experimental feature.
If you see this, it's either because you selected multiple layers, or because
you enabled the experimental feature for single layer dynamic controls
(see Preferences -> Experimental -> Generate GUI layer controls dynamically).
Expect some issues, such as controls being out of sync with the layer models!
For any issues you encounter, please head to the napari repository to report,
and stay tuned for the fixes in the upcoming releases!"""


class QtDynamicLayerControls(QFrame):
    """Superclass for all the other LayerControl classes.

    This class is never directly instantiated anywhere.
    Parameters
    ----------
    layers : list of napari.layers.Layer
        A list of napari layers.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.

    layers : list of napari.layers.Layer
        A list of selected layers in the viewer
    """

    MODE = Mode
    # PAN_ZOOM_ACTION_NAME = ''
    # TRANSFORM_ACTION_NAME = ''

    def __init__(self, layers) -> None:
        super().__init__()

        self._ndisplay: int = 2
        self._layers = layers

        self.setObjectName('layer')
        self.setMouseTracking(True)

        layout = LayerFormLayout(self)
        self.setLayout(layout)

        if len(layers) == 1:
            for layer_type, buttons_class in buttons_dict.items():
                if isinstance(layers[0], layer_type):
                    self.buttons = buttons_class(layers[0])
        else:
            self.buttons = QtMultiLayerButtons(layers[0])
        layout.addRow(self.buttons)

        for layer_type, controls in controls_dict.items():
            if all(isinstance(layer, layer_type) for layer in self._layers):
                for control in controls:
                    if control is QtHistogramControl and len(self._layers) > 1:
                        continue
                    self._add_widget_controls(
                        control(parent=self, layers=layers)
                    )
        for layer in self._layers:
            layer.events.data.connect(self._on_surface_coloring_change)

        # warning experimental: qss theme takes care of the icon by giving it a name
        warn_icon = QLabel()
        warn_icon.setObjectName('error_label')
        warn_icon.setToolTip(EXPERIMENTAL_WARNING)
        # need a separate container widget for centering to work nice
        warn_widget = QWidget(self)
        warn_layout = QHBoxLayout(warn_widget)
        warn_layout.setContentsMargins(0, 0, 0, 0)
        warn_layout.addStretch(1)
        warn_layout.addWidget(warn_icon)
        warn_layout.addStretch(1)
        warn_widget.setAttribute(Qt.WA_TranslucentBackground)
        layout.addRow('experimental!', warn_widget)
        self._on_surface_coloring_change()
        self._on_ndisplay_changed()

    def _add_widget_controls(
        self,
        wrapper: QtWidgetControlsBase,
    ) -> None:
        """
        Add widget controls.

        Parameters
        ----------
        wrapper : napari._qt.layer_controls.dynamic..widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        """
        controls = wrapper.get_widget_controls()

        for label_text, control_widget in controls:
            self.layout().addRow(label_text, control_widget)

    @property
    def ndisplay(self) -> int:
        """The number of dimensions displayed in the canvas."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only like the transform mode button.
        """
        depiction = self.findChild(QtDepictionControl)
        if depiction is not None:
            depiction._change_ndisplay(self._ndisplay)

        interpolation = self.findChild(QtInterpolationComboBoxControl)
        if interpolation is not None:
            interpolation._update_interpolation_combo(self.ndisplay)

        rendering_image = self.findChild(QtImageRenderControl)
        if rendering_image is not None:
            rendering_image._change_ndisplay(self._ndisplay)

        rendering_labels = self.findChild(QtLabelRenderingControl)
        if rendering_labels is not None:
            rendering_labels._change_ndisplay(self._ndisplay)

        label_buttons = self.findChild(QtLabelsButtons)
        if label_buttons is not None:
            label_buttons._set_polygon_tool_state()

        buttons = self.findChild(QtLayerButtons)
        if buttons is not None:
            buttons.ndisplay = self.ndisplay

    def _on_surface_coloring_change(
        self,
    ) -> None:
        """Disable scalar-color controls when direct vertex colors are active."""
        enabled = all(
            getattr(layer, 'vertex_colors', None) is None
            for layer in self._layers
        )
        for cls in (
            QtContrastLimitsControl,
            QtGammaSliderControl,
            QtColormapControl,
        ):
            control = self.findChild(cls)
            if control is None:
                continue
            for label, widget in control.get_widget_controls():
                set_widgets_enabled_with_opacity(
                    self, (label, widget), enabled
                )

    def _disconnect_child_widget_controls(self, child) -> None:
        disconnect_method = getattr(child, 'disconnect_widget_controls', None)
        if disconnect_method is not None:
            disconnect_method()

    def deleteLater(self):
        disconnect_events(self._layers[0].events, self.buttons)
        for layer in self._layers:
            disconnect_events(layer.events, self)
        for child in self.children():
            self._disconnect_child_widget_controls(child)
        super().deleteLater()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self._layers[0].events, self.buttons)
        for layer in self._layers:
            disconnect_events(layer.events, self)
        for child in self.children():
            self._disconnect_child_widget_controls(child)
            getattr(child, 'close', lambda: None)()
        super().close()
