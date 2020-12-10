from qtpy.QtWidgets import QFrame, QStackedWidget

from ...layers import Image, Labels, Points, Shapes, Surface, Tracks, Vectors
from ...utils import config
from .qt_image_controls import QtImageControls
from .qt_labels_controls import QtLabelsControls
from .qt_points_controls import QtPointsControls
from .qt_shapes_controls import QtShapesControls
from .qt_surface_controls import QtSurfaceControls
from .qt_tracks_controls import QtTracksControls
from .qt_vectors_controls import QtVectorsControls

layer_to_controls = {
    Labels: QtLabelsControls,
    Image: QtImageControls,  # must be after Labels layer
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Vectors: QtVectorsControls,
    Tracks: QtTracksControls,
}

if config.async_loading:
    from ...layers.image.experimental.octree_image import OctreeImage

    # The user visible layer controls for OctreeImage layers are identical
    # to the regular image layer controls, for now.
    layer_to_controls[OctreeImage] = QtImageControls


def create_qt_layer_controls(layer):
    """
    Create a qt controls widget for a layer based on its layer type.

    Parameters
    ----------
    layer : napari.layers._base_layer.Layer
        Layer that needs its controls widget created.

    Returns
    -------
    controls : napari.layers.base.QtLayerControls
        Qt controls widget
    """

    for layer_type, controls in layer_to_controls.items():
        if isinstance(layer, layer_type):
            return controls(layer)

    raise TypeError(
        f'Could not find QtControls for layer of type {type(layer)}'
    )


class QtLayerControlsContainer(QStackedWidget):
    """Container widget for QtLayerControl widgets.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    empty_widget : qtpy.QtWidgets.QFrame
        Empty placeholder frame for when no layer is selected.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    widgets : dict
        Dictionary of key value pairs matching layer with its widget controls.
        widgets[layer] = controls
    """

    def __init__(self, viewer):
        super().__init__()
        self.setProperty("emphasized", True)
        self.viewer = viewer

        self.setMouseTracking(True)
        self.empty_widget = QFrame()
        self.widgets = {}
        self.addWidget(self.empty_widget)
        self._display(None)

        self.viewer.layers.events.inserted.connect(self._add)
        self.viewer.layers.events.removed.connect(self._remove)
        self.viewer.events.active_layer.connect(self._display)

    def _display(self, event):
        """Change the displayed controls to be those of the target layer.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        if event is None:
            layer = None
        else:
            layer = event.item

        if layer is None:
            self.setCurrentWidget(self.empty_widget)
        else:
            controls = self.widgets[layer]
            self.setCurrentWidget(controls)

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        layer = event.value
        controls = create_qt_layer_controls(layer)
        self.addWidget(controls)
        self.widgets[layer] = controls

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        layer = event.value
        controls = self.widgets[layer]
        self.removeWidget(controls)
        controls.close()
        controls = None
        del self.widgets[layer]
