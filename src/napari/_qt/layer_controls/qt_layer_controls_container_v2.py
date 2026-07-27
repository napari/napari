from qtpy.QtWidgets import QFrame, QStackedWidget
from napari._qt.layer_controls.qt_dynamic_layer_controls import QtDynamicLayerControls

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari._qt.layer_controls.qt_points_controls import QtPointsControls
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari._qt.layer_controls.qt_surface_controls import QtSurfaceControls
from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari._qt.layer_controls.qt_vectors_controls import QtVectorsControls
from napari.layers import (
    Image,
    Labels,
    Points,
    Shapes,
    Surface,
    Tracks,
    Vectors,
)

layer_to_controls = {
    Labels: QtLabelsControls,
    Image: QtImageControls,
    Points: QtPointsControls,
    Shapes: QtShapesControls,
    Surface: QtSurfaceControls,
    Vectors: QtVectorsControls,
    Tracks: QtTracksControls,
}



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

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer

        self.setMouseTracking(True)
        self.empty_widget = QFrame()
        self.empty_widget.setObjectName('empty_controls_widget')
        self.panel = None
        self.addWidget(self.empty_widget)
        self.setCurrentWidget(self.empty_widget)

        viewer.layers.selection.events.changed.connect(self._populate)
        viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
        viewer.events.theme.connect(self._on_viewer_theme_changed)

    def _on_ndisplay_changed(self, event):
        """Responds to a change in the dimensionality displayed in the canvas.

        Parameters
        ----------
        event : Event
            Event with the new dimensionality value at `event.value`.
        """
        for widget in self.panel.values():
            if widget is not self.empty_widget:
                widget.ndisplay = event.value

    def _on_viewer_theme_changed(self, event=None):
        """Respond to viewer.theme changes from keybindings (Ctrl+Shift+T).

        The ``toggle_theme`` keybinding sets ``viewer.theme`` directly
        without updating ``settings.appearance.theme``, so widgets that
        listen only to settings events miss the change. This bridges
        the gap by forwarding ``event.value`` (the new theme) to any
        histogram widgets that have been lazily created.
        """
        for widget in self.panel.values():
            histogram_control = getattr(widget, '_histogram_control', None)
            if histogram_control is None:
                continue
            hist_widget = getattr(histogram_control, 'histogram_widget', None)
            if hist_widget is not None:
                hist_widget._on_theme_change(event)

    def _populate(self, event):
        """Change the displayed controls to be those of the target layers.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        selection = self.viewer.layers.selection
        layers = [l for l in self.viewer if l in selection]
        self._remove_controls()
        if not layers:
            self.setCurrentWidget(self.empty_widget)
        else:
            self.panel = QtDynamicLayerControls(layers)
            self.panel.ndisplay = self.viewer.dims.ndisplay
            self.addWidget(self.panel)
            self.setCurrentWidget(self.panel)

    def _remove(self, event):
        """Remove the controls target layer from the list of control widgets.

        Parameters
        ----------
        event : Event
            Event with the target layer at `event.value`.
        """
        self.removeWidget(self.panel)
        self.panel.hide()
        self.panel.deleteLater()
        self.panel = None
