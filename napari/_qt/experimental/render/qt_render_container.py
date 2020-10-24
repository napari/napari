"""QtRenderContainer class.
"""


from qtpy.QtWidgets import QStackedWidget

from .qt_render import QtRender


class QtRenderContainer(QStackedWidget):
    """Container widget for QtRender widgets.

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

        # We show QtRender even when there is no layer. However in that
        # case it only shows the "create test image" controls, the controls
        # that can be used to create a new layer.
        self.empty_widget = QtRender(viewer)

        self._widgets = {}
        self.addWidget(self.empty_widget)
        self._display(None)

        self.viewer.layers.events.added.connect(self._add)
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
            controls = self._widgets[layer]
            self.setCurrentWidget(controls)

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        layer = event.item
        controls = QtRender(self.viewer, layer)
        self.addWidget(controls)
        self._widgets[layer] = controls

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        layer = event.item
        controls = self._widgets[layer]
        self.removeWidget(controls)
        controls.deleteLater()
        controls = None
        del self._widgets[layer]
