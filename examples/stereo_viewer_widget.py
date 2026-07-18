"""
Stereo 3D viewer widget
=======================

Two side-by-side 3D viewers in a standalone window, with synchronized
layers and cameras. A small horizontal eye separation is applied so the
pair can be viewed stereoscopically (parallel / wall-eyed fusion).

Interact with either eye view or the main viewer: angles, zoom, and
perspective stay locked; only the look-at center is offset left/right.

Use the eye-separation spinbox to tune parallax for your data and
screen size. Cross-eyed viewing: swap which pane is left vs right, or
use a negative separation. If the stereo window is closed, reopen it
from the "Stereo" dock button in the main viewer.

Based on the layer-sync pattern in ``multiple_viewer_widget.py``.

.. tags:: gui, visualization-nD
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import napari
from napari.components.camera import Camera
from napari.components.viewer_model import ViewerModel
from napari.layers import Labels, Layer
from napari.qt import QtViewer
from napari.utils.events.event import WarningEmitter

def copy_layer(layer: Layer, name: str = ''):
    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata['viewer_name'] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ('thumbnail', 'name'):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


def _camera_right_vector(angles: tuple[float, float, float]) -> np.ndarray:
    """Unit vector pointing right on the canvas for the given Euler angles."""
    cam = Camera(angles=angles)
    right = np.cross(cam.up_direction, cam.view_direction)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return right / norm


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str | None = None,
        layer_type: str | None = None,
        **kwargs,
    ):
        """Forward drag-and-drop opens to the main viewer."""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class StereoViewerWidget(QWidget):
    """Side-by-side 3D viewers with a stereo camera offset."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_left = ViewerModel(title='left eye')
        self.viewer_right = ViewerModel(title='right eye')
        self._block = False
        self._eye_separation = 20.0

        self.qt_left = QtViewerWrap(viewer, self.viewer_left)
        self.qt_right = QtViewerWrap(viewer, self.viewer_right)

        self.viewer.dims.ndisplay = 3
        self.viewer_left.dims.ndisplay = 3
        self.viewer_right.dims.ndisplay = 3

        controls = QHBoxLayout()

        # eye separation control
        controls.addWidget(QLabel('Eye separation'))
        self.separation_spin = QDoubleSpinBox()
        self.separation_spin.setRange(-500.0, 500.0)
        self.separation_spin.setSingleStep(1.0)
        self.separation_spin.setDecimals(1)
        self.separation_spin.setValue(self._eye_separation)
        self.separation_spin.valueChanged.connect(self._on_separation_changed)
        controls.addWidget(self.separation_spin)
        controls.addStretch(1)

        # place the left and right viewers side by side
        eye_layout = QHBoxLayout()
        eye_layout.addWidget(self.qt_left)
        eye_layout.addWidget(self.qt_right)

        # layout the controls and the viewers
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(eye_layout)
        self.setLayout(layout)

        # listen to layer events
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_left.dims.events.current_step.connect(self._point_update)
        self.viewer_right.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.ndisplay.connect(self._ndisplay_update)
        self.viewer.events.reset_view.connect(self._reset_view)

        # listen to camera events
        for model in self._all_models:
            model.camera.events.angles.connect(self._camera_update)
            model.camera.events.center.connect(self._camera_update)
            model.camera.events.zoom.connect(self._camera_update)
            model.camera.events.perspective.connect(self._camera_update)

        # listen to status events
        self.viewer_left.events.status.connect(self._status_update)
        self.viewer_right.events.status.connect(self._status_update)

        # Mild perspective helps stereo depth cues
        self.viewer.camera.perspective = 30

    @property
    def _all_models(self):
        return (self.viewer, self.viewer_left, self.viewer_right)

    @property
    def _eye_models(self):
        return (self.viewer_left, self.viewer_right)

    def _status_update(self, event):
        self.viewer.status = event.value

    def _resync_from_main_camera(self):
        """Re-derive both eye views from the main viewer's camera."""
        cam = self.viewer.camera
        self._apply_stereo_from(
            cam.angles, cam.center, cam.zoom, cam.perspective
        )

    def _on_separation_changed(self, value: float):
        self._eye_separation = value
        self._resync_from_main_camera()

    def _ndisplay_update(self, event):
        """Keep the stereo pair in 3D when the main viewer is in 3D."""
        if event.value != 3:
            return
        self.viewer_left.dims.ndisplay = 3
        self.viewer_right.dims.ndisplay = 3

    def _reset_view(self):
        self.viewer_left.reset_view()
        self.viewer_right.reset_view()
        self._resync_from_main_camera()

    def _base_center_from(self, model, angles, center) -> np.ndarray:
        """Undo eye offset to recover the shared (cyclopean) look-at."""
        right = _camera_right_vector(angles)
        half = self._eye_separation / 2.0
        center = np.asarray(center, dtype=float)
        if model is self.viewer_left:
            return center + half * right
        if model is self.viewer_right:
            return center - half * right
        return center

    def _apply_stereo_from(self, angles, center, zoom, perspective):
        """Apply shared camera state with left/right look-at offsets."""
        right = _camera_right_vector(tuple(angles))
        half = self._eye_separation / 2.0
        base = np.asarray(center, dtype=float)
        try:
            self._block = True
            for model, sign in (
                (self.viewer, 0.0),
                (self.viewer_left, -1.0),
                (self.viewer_right, 1.0),
            ):
                model.dims.ndisplay = 3
                model.camera.angles = tuple(angles)
                model.camera.zoom = zoom
                model.camera.perspective = perspective
                model.camera.center = tuple(base + sign * half * right)
        finally:
            self._block = False

    def _camera_update(self, event):
        if self._block:
            return
        source_cam = event.source
        model = next(m for m in self._all_models if m.camera is source_cam)
        base = self._base_center_from(
            model, source_cam.angles, source_cam.center
        )
        self._apply_stereo_from(
            source_cam.angles,
            base,
            source_cam.zoom,
            source_cam.perspective,
        )

    def _layer_selection_changed(self, event):
        """Update the active layer in the eye viewers."""
        if self._block:
            return
        if event.value is None:
            self.viewer_left.layers.selection.active = None
            self.viewer_right.layers.selection.active = None
            return
        self.viewer_left.layers.selection.active = self.viewer_left.layers[
            event.value.name
        ]
        self.viewer_right.layers.selection.active = self.viewer_right.layers[
            event.value.name
        ]

    def _point_update(self, event):
        for model in self._all_models:
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _layer_added(self, event):
        """Add the layer to both eye viewers and connect sync events."""
        self.viewer_left.layers.insert(
            event.index, copy_layer(event.value, 'left')
        )
        self.viewer_right.layers.insert(
            event.index, copy_layer(event.value, 'right')
        )
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(self._property_sync)

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            event.value.events.labels_update.connect(self._set_data_refresh)
            for eye in self._eye_models:
                eye.layers[event.value.name].events.set_data.connect(
                    self._set_data_refresh
                )
                eye.layers[event.value.name].events.labels_update.connect(
                    self._set_data_refresh
                )

        self.viewer_left.layers[event.value.name].events.data.connect(
            self._sync_data
        )
        self.viewer_right.layers[event.value.name].events.data.connect(
            self._sync_data
        )
        event.value.events.name.connect(self._sync_name)

    def _sync_name(self, event):
        """Sync layer names across viewers."""
        index = self.viewer.layers.index(event.source)
        self.viewer_left.layers[index].name = event.source.name
        self.viewer_right.layers[index].name = event.source.name

    def _sync_data(self, event):
        """Sync data modifications from an eye viewer back to the others."""
        if self._block:
            return
        for model in self._all_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """Synchronize Labels refresh between viewers."""
        if self._block:
            return
        for model in self._all_models:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """Remove the layer from both eye viewers."""
        self.viewer_left.layers.pop(event.index)
        self.viewer_right.layers.pop(event.index)

    def _layer_moved(self, event):
        """Update layer order in both eye viewers."""
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_left.layers.move(event.index, dest_index)
        self.viewer_right.layers.move(event.index, dest_index)

    def _property_sync(self, event):
        """Sync layer properties (except name) to both eye viewers.

        Property name comes from ``event.type`` (e.g. ``'opacity'``), so we
        can connect this method directly without ``functools.partial``.
        """
        if event.source not in self.viewer.layers:
            return
        name = event.type
        try:
            self._block = True
            value = getattr(event.source, name)
            setattr(self.viewer_left.layers[event.source.name], name, value)
            setattr(self.viewer_right.layers[event.source.name], name, value)
        finally:
            self._block = False


class StereoViewerWindow(QMainWindow):
    """Standalone window hosting the left/right stereo viewers."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.setWindowTitle('Stereo 3D')
        # Keep the window alive when closed so it can be re-opened
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.stereo = StereoViewerWidget(viewer)
        self.setCentralWidget(self.stereo)
        self.resize(1000, 500)

    def closeEvent(self, event):
        self.hide()
        event.accept()


class StereoOpenButton(QWidget):
    """Dock control to (re)open the stereo window."""

    def __init__(self, stereo_window: StereoViewerWindow) -> None:
        super().__init__()
        self._stereo_window = stereo_window
        btn = QPushButton('Open stereo window')
        btn.clicked.connect(self._open)
        layout = QVBoxLayout()
        layout.addWidget(btn)
        layout.addStretch(1)
        self.setLayout(layout)

    def _open(self):
        self._stereo_window.show()
        self._stereo_window.raise_()
        self._stereo_window.activateWindow()


if __name__ == '__main__':


    # Needed so additional OpenGL canvases can share contexts with the main viewer
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )

    view = napari.Viewer(ndisplay=3)
    stereo_window = StereoViewerWindow(view)
    stereo_window.show()

    # Dock widget holds a reference to stereo_window (via StereoOpenButton)
    view.window.add_dock_widget(
        StereoOpenButton(stereo_window),
        name='Stereo',
        area='left',
    )

    view.open_sample('napari', 'cells3d')
    view.camera.angles = (-20, 20, -20)

    napari.run()
