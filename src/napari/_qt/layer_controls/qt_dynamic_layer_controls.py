# TODO: sort through necessary imports
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
)

from napari._qt.layer_controls.qt_image_buttons import QtImageButtons
from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_buttons import QtLabelsButtons
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.qt_points_buttons import QtPointsButtons
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.qt_shapes_buttons import QtShapesButtons
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari._qt.layer_controls.qt_surface_buttons import QtSurfaceButtons
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_tracks_buttons import QtTracksButtons
from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari._qt.layer_controls.qt_vectors_buttons import QtVectorsButtons
from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari._qt.layer_controls.widgets import (
    QtOpacityBlendingControls,
    QtWidgetControlsBase,
)
from napari._qt.layer_controls.widgets._points.qt_border_color import (
    QtBorderColorControl,
)
from napari._qt.layer_controls.widgets.qt_colormap_control import (
    QtColormapControl,
)
from napari._qt.layer_controls.widgets.qt_contrast_limits import (
    QtContrastLimitsControl,
)
from napari._qt.layer_controls.widgets.qt_face_color import QtFaceColorControl
from napari._qt.layer_controls.widgets.qt_gamma_slider import (
    QtGammaSliderControl,
)
from napari._qt.layer_controls.widgets.qt_histogram_control import (
    QtHistogramControl,
)
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)
from napari.layers.base._base_constants import Mode
from napari.utils.events import disconnect_events

layer_to_controls = {
    Labels: QtLabelsControls,
    Image: QtImageControls,
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Vectors: QtVectorsControls,
    Tracks: QtTracksControls,
}

controls_dict = {
    'opacity': QtOpacityBlendingControls,
    'contrast_limits': QtContrastLimitsControl,
    'histogram': QtHistogramControl,
    'gamma': QtGammaSliderControl,
    'colormap': QtColormapControl,
    'face_color': QtFaceColorControl,
    'border_color': QtBorderColorControl,
    #'projection_mode': QtProjectionModeControl,
    #'interpolation': QtInterpolationComboBoxControl,
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

    def __init__(self, QWidget=None) -> None:
        super().__init__(QWidget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)


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
    _opacity_blending_controls: napari._qt.layer_controls.widgets.QtOpacityBlendingControls
        Wrapper widget with a dropdown widget to select the layer blending mode and
        a slider for the layer opacity.
    layer : napari.layers.Layer
        An instance of a napari layer.
    layers : list of selected layers in the viewer
    """

    MODE = Mode
    # PAN_ZOOM_ACTION_NAME = ''
    # TRANSFORM_ACTION_NAME = ''

    def __init__(self, layers) -> None:
        super().__init__()

        self._ndisplay: int = 2
        # self._EDIT_BUTTONS: tuple = ()
        # self._MODE_BUTTONS: dict = {}
        self._layers = layers

        # for layer in self._layers:
        # self.layer.events.mode.connect(self._on_mode_change)
        # self.layer.events.editable.connect(self._on_editable_or_visible_change)
        # self.layer.events.visible.connect(self._on_editable_or_visible_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.setLayout(LayerFormLayout(self))

        for layer_type, buttons in buttons_dict.items():
            if len(layers) == 1 and isinstance(layers[0], layer_type):
                self.layout().addRow(buttons(layers[0]))

        # @lorenzo does this go into a seperate function or the init? Also do I need to
        # add the controls_dict anywhere
        for attr, control in controls_dict.items():
            if all(hasattr(layer, attr) for layer in self._layers):
                self._add_widget_controls(control(parent=self, layers=layers))

    def _add_widget_controls(
        self,
        wrapper: QtWidgetControlsBase,
    ) -> None:
        """
        Add widget controls.

        Parameters
        ----------
        wrapper : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        """
        controls = wrapper.get_widget_controls()

        for label_text, control_widget in controls:
            self.layout().addRow(label_text, control_widget)

    def changeProjectionMode(self, text):
        for layer in self._layers:
            with layer.events.blocker(self._on_projection_mode_change):
                layer.projection_mode = text

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

    def _disconnect_child_widget_controls(self, child) -> None:
        disconnect_method = getattr(child, 'disconnect_widget_controls', None)
        if disconnect_method is not None:
            disconnect_method()

    def deleteLater(self):
        for layer in self._layers:
            disconnect_events(layer.events, self)
            for child in self.children():
                self._disconnect_child_widget_controls(child)
            super().deleteLater()

    def close(self):
        """Disconnect events when widget is closing."""
        for layer in self._layers:
            disconnect_events(layer.events, self)
            for child in self.children():
                close_method = getattr(child, 'close', None)
                self._disconnect_child_widget_controls(child)
                if close_method is not None:
                    close_method()
            return super().close()
