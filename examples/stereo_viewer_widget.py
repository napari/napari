"""
Stereo 3D viewer widget
=======================

Two side-by-side 3D viewers with synchronized layers and cameras.
A small horizontal eye separation is applied so the pair can be viewed
stereoscopically (parallel / wall-eyed fusion).

Interact with either eye view or the main viewer: angles, zoom, and
perspective stay locked; only the look-at center is offset left/right.

Use the eye-separation spinbox to tune parallax for your data and
screen size. Cross-eyed viewing: swap which pane is left vs right, or
use a negative separation.

.. tags:: gui, visualization-nD
"""

from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import napari
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
    from napari.components.camera import Camera

    cam = Camera(angles=angles)
    right = np.cross(cam.up_direction, cam.view_direction)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return right / norm


class own_partial:
    """Workaround for deepcopy not copying partial functions."""

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


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
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class StereoViewerWidget(QWidget):
    """Side-by-side 3D viewers with stereo camera offset."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_left = ViewerModel(title='left eye')
        self.viewer_right = ViewerModel(title='right eye')
        self._block = False
        self._eye_separation = 20.0

        self.qt_left = QtViewerWrap(viewer, self.viewer_left)
        self.qt_right = QtViewerWrap(viewer, self.viewer_right)

        # Force 3D on all viewers
        self.viewer.dims.ndisplay = 3
        self.viewer_left.dims.ndisplay = 3
        self.viewer_right.dims.ndisplay = 3

        controls = QHBoxLayout()
        controls.addWidget(QLabel('Eye separation'))
        self.separation_spin = QDoubleSpinBox()
        self.separation_spin.setRange(-500.0, 500.0)
        self.separation_spin.setSingleStep(1.0)
        self.separation_spin.setDecimals(1)
        self.separation_spin.setValue(self._eye_separation)
        self.separation_spin.valueChanged.connect(self._on_separation_changed)
        controls.addWidget(self.separation_spin)
        controls.addStretch(1)

        eye_splitter = QSplitter()
        eye_splitter.setOrientation(Qt.Orientation.Horizontal)
        eye_splitter.addWidget(self.qt_left)
        eye_splitter.addWidget(self.qt_right)
        eye_splitter.setSizes([400, 400])

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(eye_splitter)
        self.setLayout(layout)

        # Layer sync
        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )

        # Dims sync
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_left.dims.events.current_step.connect(self._point_update)
        self.viewer_right.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.ndisplay.connect(self._ndisplay_update)
        self.viewer.events.reset_view.connect(self._reset_view)

        # Camera sync (stereo)
        for model in (self.viewer, self.viewer_left, self.viewer_right):
            model.camera.events.angles.connect(self._camera_update)
            model.camera.events.center.connect(self._camera_update)
            model.camera.events.zoom.connect(self._camera_update)
            model.camera.events.perspective.connect(self._camera_update)

        self.viewer_left.events.status.connect(self._status_update)
        self.viewer_right.events.status.connect(self._status_update)

        # Mild perspective helps stereo depth cues
        self.viewer.camera.perspective = 30

    def _status_update(self, event):
        self.viewer.status = event.value

    def _on_separation_changed(self, value: float):
        self._eye_separation = value
        self._apply_stereo_from(
            self.viewer.camera.angles,
            self.viewer.camera.center,
            self.viewer.camera.zoom,
            self.viewer.camera.perspective,
        )

    def _ndisplay_update(self, event):
        # Keep the stereo pair in 3D even if the main viewer flips to 2D
        if event.value != 3:
            return
        self.viewer_left.dims.ndisplay = 3
        self.viewer_right.dims.ndisplay = 3

    def _reset_view(self):
        self.viewer_left.reset_view()
        self.viewer_right.reset_view()
        self._apply_stereo_from(
            self.viewer.camera.angles,
            self.viewer.camera.center,
            self.viewer.camera.zoom,
            self.viewer.camera.perspective,
        )

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
        model = next(
            m
            for m in (self.viewer, self.viewer_left, self.viewer_right)
            if m.camera is source_cam
        )
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
        for model in (self.viewer, self.viewer_left, self.viewer_right):
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _layer_added(self, event):
        self.viewer_left.layers.insert(
            event.index, copy_layer(event.value, 'left')
        )
        self.viewer_right.layers.insert(
            event.index, copy_layer(event.value, 'right')
        )
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._set_data_refresh)
            event.value.events.labels_update.connect(self._set_data_refresh)
            for eye in (self.viewer_left, self.viewer_right):
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
        index = self.viewer.layers.index(event.source)
        self.viewer_left.layers[index].name = event.source.name
        self.viewer_right.layers[index].name = event.source.name

    def _sync_data(self, event):
        if self._block:
            return
        for model in (self.viewer, self.viewer_left, self.viewer_right):
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        if self._block:
            return
        for model in (self.viewer, self.viewer_left, self.viewer_right):
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        self.viewer_left.layers.pop(event.index)
        self.viewer_right.layers.pop(event.index)

    def _layer_moved(self, event):
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_left.layers.move(event.index, dest_index)
        self.viewer_right.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            value = getattr(event.source, name)
            setattr(self.viewer_left.layers[event.source.name], name, value)
            setattr(self.viewer_right.layers[event.source.name], name, value)
        finally:
            self._block = False


if __name__ == '__main__':
    from qtpy import QtWidgets

    QtWidgets.QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )

    view = napari.Viewer(ndisplay=3)
    stereo = StereoViewerWidget(view)
    view.window.add_dock_widget(stereo, name='Stereo 3D', area='bottom')

    view.open_sample('napari', 'cells3d')
    view.camera.angles = (-20, 20, -20)

    napari.run()
