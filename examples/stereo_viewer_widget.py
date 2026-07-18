"""
Stereo 3D viewer
================

Two side-by-side 3D viewers with the same layers and synchronized cameras.
A small horizontal eye separation is applied so the pair can be viewed
stereoscopically (parallel / wall-eyed fusion).

Interact with either eye view or the main viewer: angles, zoom, and
perspective stay locked; only the look-at center is offset left/right.

Use the eye-separation spinbox to tune parallax for your data and
screen size. Cross-eyed viewing: use a negative separation.

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
    QVBoxLayout,
    QWidget,
)

import napari
from napari.components.camera import Camera
from napari.components.viewer_model import ViewerModel
from napari.layers import Layer
from napari.qt import QtViewer


def copy_layer(layer: Layer) -> Layer:
    return Layer.create(*layer.as_layer_data_tuple())


class StereoViewerWidget(QWidget):
    """Side-by-side 3D viewers with a stereo camera offset."""

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.viewer_left = ViewerModel(title='left eye')
        self.viewer_right = ViewerModel(title='right eye')
        self._block = False
        self._eye_separation = 20.0
        # Reused only to derive view/up directions from Euler angles.
        self._direction_camera = Camera()

        # Create viewers first, then add layers
        self.qt_left = QtViewer(self.viewer_left)
        self.qt_right = QtViewer(self.viewer_right)

        for layer in viewer.layers:
            self.viewer_left.layers.append(copy_layer(layer))
            self.viewer_right.layers.append(copy_layer(layer))

        for model in self._all_models():
            model.dims.ndisplay = 3

        # eye separation control
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

        # left and right eye viewers
        eye_layout = QHBoxLayout()
        eye_layout.addWidget(self.qt_left)
        eye_layout.addWidget(self.qt_right)

        # main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(eye_layout)
        self.setLayout(layout)

        for model in self._all_models():
            model.camera.events.angles.connect(self._camera_update)
            model.camera.events.center.connect(self._camera_update)
            model.camera.events.zoom.connect(self._camera_update)
            model.camera.events.perspective.connect(self._camera_update)

        self.viewer.events.reset_view.connect(self._reset_view)
        # Mild perspective helps stereo depth cues.
        self.viewer.camera.perspective = 30

    def _all_models(self) -> tuple[ViewerModel, ViewerModel, ViewerModel]:
        return (self.viewer, self.viewer_left, self.viewer_right)

    def _camera_right_vector(
        self, angles: tuple[float, float, float]
    ) -> np.ndarray:
        """Unit vector pointing right on the canvas for the given Euler angles."""
        self._direction_camera.angles = angles
        right = np.cross(
            self._direction_camera.up_direction,
            self._direction_camera.view_direction,
        )
        norm = np.linalg.norm(right)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])
        return right / norm

    def _on_separation_changed(self, value: float) -> None:
        # when the eye separation is changed, re-calculate and apply camera state to all viewers
        self._eye_separation = value
        cam = self.viewer.camera
        self._apply_stereo_from(
            cam.angles, cam.center, cam.zoom, cam.perspective
        )

    def _reset_view(self) -> None:
        # Main viewer already reset; push its camera through stereo sync.
        cam = self.viewer.camera
        self._apply_stereo_from(
            cam.angles, cam.center, cam.zoom, cam.perspective
        )

    def _base_center_from(
        self,
        model: ViewerModel,
        angles: tuple[float, float, float],
        center: tuple[float, float, float] | tuple[float, float],
    ) -> np.ndarray:
        """Shared (cyclopean) look-at center implied by an eye camera center."""
        right_direction = self._camera_right_vector(angles)
        half = self._eye_separation / 2.0
        center_arr = np.asarray(center, dtype=float)
        if model is self.viewer_left:
            return center_arr + half * right_direction
        if model is self.viewer_right:
            return center_arr - half * right_direction
        return center_arr

    def _apply_stereo_from(
        self,
        angles: tuple[float, float, float],
        center: tuple[float, float, float] | tuple[float, float] | np.ndarray,
        zoom: float,
        perspective: float,
    ) -> None:
        """Apply shared camera state with left/right look-at offsets."""
        right_direction = self._camera_right_vector(angles)
        half = self._eye_separation / 2.0
        base = np.asarray(center, dtype=float)
        prev = self._block
        self._block = True
        try:
            for model, sign in (
                (self.viewer, 0.0),
                (self.viewer_left, -1.0),
                (self.viewer_right, 1.0),
            ):
                model.camera.angles = angles
                model.camera.zoom = zoom
                model.camera.perspective = perspective
                model.camera.center = tuple(
                    base + sign * half * right_direction
                )
        finally:
            self._block = prev

    def _camera_update(self, event) -> None:
        if self._block:
            return
        source_cam = event.source
        model = next(m for m in self._all_models() if m.camera is source_cam)
        # shared look-at without this eye's stereo offset
        base = self._base_center_from(
            model, source_cam.angles, source_cam.center
        )
        # apply the stereo effect to the viewer
        self._apply_stereo_from(
            source_cam.angles,
            base,
            source_cam.zoom,
            source_cam.perspective,
        )


if __name__ == '__main__':
    # Needed so additional OpenGL canvases can share contexts with the main viewer.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    viewer = napari.Viewer(ndisplay=3)
    viewer.open_sample('napari', 'cells3d')
    viewer.camera.angles = (-20, 20, -20)

    window = QMainWindow()
    window.setWindowTitle('Stereo 3D')
    window.setCentralWidget(StereoViewerWidget(viewer))
    window.resize(1000, 500)
    window.show()

    napari.run()
