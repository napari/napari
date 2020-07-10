from qtpy.QtWidgets import QStackedWidget, QFrame
from .layers import create_qt_controls


class QtControls(QStackedWidget):
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
            controls = self.widgets[layer]
            self.setCurrentWidget(controls)

    def _add(self, event):
        """Add the controls target layer to the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        layer = event.item
        controls = create_qt_controls(layer)
        self.addWidget(controls)
        self.widgets[layer] = controls

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.item`.
        """
        layer = event.item
        controls = self.widgets[layer]
        self.removeWidget(controls)
        controls.deleteLater()
        controls = None
        del self.widgets[layer]
