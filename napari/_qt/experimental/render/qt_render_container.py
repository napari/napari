"""QtRenderContainer class.
"""
from qtpy.QtWidgets import QStackedWidget

from ....components.viewer_model import ViewerModel
from .qt_render import QtRender


class QtRenderContainer(QStackedWidget):
    """Container widget for QtRender widgets.

    QtRender is a debug/developer widget for rendering and octree related
    functionality. We put up a QtRender for any later, but most of the
    controls are only visible for OctreeImage layers.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    default_widget : QtRender
        A minimal version fo QtRender if no layer is selected.
    viewer : ViewerModel
        Napari viewer.
    widgets : dict
        Maps layer to its QtRender widget.
    """

    def __init__(self, viewer: ViewerModel):

        super().__init__()
        self.setProperty("emphasized", True)
        self.viewer = viewer

        self.setMouseTracking(True)
        self.setMinimumWidth(250)

        # We show QtRender even if no layer is selected. But in that case the
        # only control are to create test images.
        self.default_widget = QtRender(viewer)

        self._widgets = {}
        self.addWidget(self.default_widget)
        self._display(None)

        self.viewer.layers.events.inserted.connect(self._add)
        self.viewer.layers.events.removed.connect(self._remove)
        self.viewer.events.active_layer.connect(self._display)

    def _display(self, event):
        """Change the displayed controls to be those of the target layer.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        if event is None:
            layer = None
        else:
            layer = event.item

        if layer is None:
            self.setCurrentWidget(self.default_widget)
        else:
            controls = self._widgets[layer]
            self.setCurrentWidget(controls)

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        layer = event.value
        controls = QtRender(self.viewer, layer)
        self.addWidget(controls)
        self._widgets[layer] = controls

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        layer = event.value
        controls = self._widgets[layer]
        self.removeWidget(controls)
        controls.deleteLater()
        controls = None
        del self._widgets[layer]
