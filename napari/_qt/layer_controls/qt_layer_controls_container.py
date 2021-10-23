from typing import TYPE_CHECKING

from qtpy.QtWidgets import QFrame, QStackedWidget

from ... import layers
from ...utils import config
from ...utils.translations import trans
from .qt_image_controls import QtImageControls
from .qt_labels_controls import QtLabelsControls
from .qt_points_controls import QtPointsControls
from .qt_shapes_controls import QtShapesControls
from .qt_surface_controls import QtSurfaceControls
from .qt_tracks_controls import QtTracksControls
from .qt_vectors_controls import QtVectorsControls

if TYPE_CHECKING:
    from ...components import LayerList

layer_to_controls = {
    layers.Labels: QtLabelsControls,
    layers.Image: QtImageControls,
    layers.Points: QtPointsControls,
    layers.Shapes: QtShapesControls,
    layers.Surface: QtSurfaceControls,
    layers.Vectors: QtVectorsControls,
    layers.Tracks: QtTracksControls,
}

if config.async_loading:
    from ...layers.image.experimental.octree_image import _OctreeImageBase

    # The user visible layer controls for OctreeImage layers are identical
    # to the regular image layer controls, for now.
    layer_to_controls[_OctreeImageBase] = QtImageControls


def create_qt_layer_controls(layer):
    """
    Create a qt controls widget for a layer based on its layer type.

    In case of a subclass, the type higher in the layer's method resolution
    order will be used.

    Parameters
    ----------
    layer : napari.layers._base_layer.Layer
        Layer that needs its controls widget created.

    Returns
    -------
    controls : napari.layers.base.QtLayerControls
        Qt controls widget
    """
    candidates = []
    for layer_type in layer_to_controls:
        if isinstance(layer, layer_type):
            candidates.append(layer_type)

    if not candidates:
        raise TypeError(
            trans._(
                'Could not find QtControls for layer of type {type_}',
                deferred=True,
                type_=type(layer),
            )
        )

    layer_cls = layer.__class__
    # Sort the list of candidates by 'lineage'
    candidates.sort(key=lambda layer_type: layer_cls.mro().index(layer_type))
    controls = layer_to_controls[candidates[0]]
    return controls(layer)


class QtLayerControlsContainer(QStackedWidget):
    """Container widget for QtLayerControl widgets.

    Parameters
    ----------
    layers : napari.components.LayerList
        list of layers in the viewer.

    Attributes
    ----------
    empty_widget : qtpy.QtWidgets.QFrame
        Empty placeholder frame for when no layer is selected.
    widgets : dict
        Dictionary of key value pairs matching layer with its widget controls.
        widgets[layer] = controls
    """

    def __init__(self, layers: 'LayerList'):
        super().__init__()
        self.setProperty("emphasized", True)
        self.setMouseTracking(True)
        self.empty_widget = QFrame()
        self.widgets = {}
        self.addWidget(self.empty_widget)

        for layer in layers:
            self._add_layer(layer)
        self.setCurrentWidget(
            self.widgets.get(layers.selection.active, self.empty_widget)
        )

        layers.events.inserted.connect(self._add)
        layers.events.removed.connect(self._remove)
        layers.selection.events.active.connect(self._display)

        @self.destroyed.connect
        def _destroy():
            layers.events.inserted.disconnect(self._add)
            layers.events.removed.disconnect(self._remove)
            layers.selection.events.active.disconnect(self._display)

    def _display(self, event):
        """Change the displayed controls to be those of the target layer.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        self.setCurrentWidget(self.widgets.get(event.value, self.empty_widget))

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        self._add_layer(event.value)

    def _add_layer(self, layer: layers.Layer):
        self.widgets[layer] = create_qt_layer_controls(layer)
        self.addWidget(self.widgets[layer])

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
        # controls.close()
        controls.hide()
        controls.deleteLater()
        controls = None
        del self.widgets[layer]
