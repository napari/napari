import numpy as np
from qtpy.QtWidgets import QWidget
from scipy.spatial.transform import Rotation
from vispy.scene import ArcballCamera, Box, Line, SceneCanvas
from vispy.visuals.transforms import MatrixTransform, STTransform

import napari


class ClippingPlanesControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._active_layer = None
        self.viewer = viewer

        self.canvas = SceneCanvas(
            keys=None, size=(600, 600), vsync=True, parent=self
        )
        self.view = self.canvas.central_widget.add_view()
        self.box = None

        self.view.camera = ArcballCamera(fov=0)
        self.view.camera.interactive = False

        self._on_extent_change()

        pos = np.array([(0, -100, -100), (0, -100, 100)])
        self.left = Line(pos, color='red', parent=self.view.scene)
        self.left.transform = STTransform()
        self.right = Line(pos, color='red', parent=self.view.scene)
        self.right.transform = STTransform()

        self.viewer.dims.events.range.connect(self._on_extent_change)
        self.viewer.dims.events.ndisplay.connect(self._on_extent_change)
        self.viewer.camera.events.angles.connect(self._on_camera_change)

        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_change
        )
        self.canvas.events.mouse_move.connect(self._on_mouse)
        self.canvas.events.mouse_press.connect(self._on_mouse)

    def _on_active_layer_change(self):
        if self._active_layer is not None:
            self._active_layer.experimental_clipping_planes.events.disconnect(
                self._on_planes_change
            )

        self._active_layer = self.viewer.layers.selection.active

        if self._active_layer is not None:
            self._active_layer.experimental_clipping_planes.events.connect(
                self._on_planes_change
            )

            self._on_planes_change()

    def _on_extent_change(self):
        displayed = self.viewer.dims.displayed
        if len(displayed) == 2:
            self.view.visible = False
        else:
            self.view.visible = True
            if self.box is not None:
                self.box.parent = None

            extents = self.viewer.layers.extent.world[:, displayed]
            sizes = extents[1] - extents[0]
            sizes /= (
                np.linalg.norm(sizes) * 50
            )  # ensure we don't have scaling issues
            self.box = Box(
                *sizes[::-1],  # swap for vispy
                face_colors=np.repeat([[0, 1, 1]], 12, axis=0),
                edge_color='black',
                parent=self.view.scene,
            )
            self.box.transform = MatrixTransform()

            self.view.camera.set_range(margin=0.5)

        self._on_camera_change()

    def _on_planes_change(self):
        pass

    def _on_camera_change(self):
        if self.box is None:
            return

        side_view = Rotation.from_euler('xz', (90, 90), degrees=True)
        camera_rot = Rotation.from_euler(
            'yzx', self.viewer.camera.angles, degrees=True
        )

        mat = np.eye(4)
        mat[:3, :3] = (side_view * camera_rot).as_matrix()
        self.box.transform.matrix = mat

    def _on_mouse(self, event=None):
        if event.type == 'mouse_press' or event.button == 1:
            p = event.pos[0] / self.canvas.size[0] / 2
            if p < 0.5:
                self.left.transform.translate = (p, 0, 0)
            else:
                self.right.transform.translate = (p, 0, 0)
