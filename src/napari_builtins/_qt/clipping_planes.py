import numpy as np
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from scipy.spatial.transform import Rotation
from superqt import QToggleSwitch
from vispy.scene import ArcballCamera, Box, InfiniteLine, SceneCanvas
from vispy.visuals.transforms import MatrixTransform, STTransform

import napari

# TODO: should be applicable to multiple layers at once if desired
# TODO: allow adding multiple and controlling direction
# TODO: allow "locking" planes, hiding them from view and leaving them there


class ClippingPlanesControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._active_layer = None
        self.viewer = viewer

        # general layout
        lay = QVBoxLayout()
        self.setLayout(lay)

        self.canvas = SceneCanvas(keys=None, size=(300, 300), vsync=True)

        self.info = QLabel(
            'This is a side view of the main napari canvas.\n'
            'Select a layer in the layerlist, then \n'
            'click/drag below to move its clipping planes.\n'
        )
        self.planes_enabled = QToggleSwitch('enabled')

        lay.addWidget(self.info)
        lay.addWidget(self.planes_enabled)
        lay.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.box = None
        self._center = np.zeros(3)

        self.view.camera = ArcballCamera(fov=0)
        self.view.camera.interactive = False
        self.camera_model = napari.components.Camera()

        self.lines = [
            InfiniteLine(pos=0, color=(1, 0, 1, 1), parent=self.view),
            InfiniteLine(pos=0, color=(1, 0, 0, 1), parent=self.view),
        ]
        self._offsets = [-1e10, 1e10]
        for line in self.lines:
            line.transform = STTransform()

        self.viewer.dims.events.range.connect(self._on_extent_change)
        self.viewer.dims.events.ndisplay.connect(self._on_extent_change)
        self.viewer.camera.events.connect(self._on_camera_change)

        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_change
        )
        self.canvas.events.mouse_move.connect(self._on_mouse)
        self.canvas.events.mouse_press.connect(self._on_mouse)
        self.planes_enabled.toggled.connect(self._update_planes)

        self._on_active_layer_change()

    def _on_active_layer_change(self):
        if self._active_layer is not None:
            self._active_layer.experimental_clipping_planes.events.disconnect(
                self._on_planes_change
            )
            self._active_layer.events.extent.disconnect(self._on_extent_change)

        self._active_layer = self.viewer.layers.selection.active

        if self._active_layer is not None:
            self._active_layer.experimental_clipping_planes.events.connect(
                self._on_planes_change
            )
            self._active_layer.events.extent.connect(self._on_extent_change)

            self._on_planes_change()

        self._on_extent_change()

    def _on_extent_change(self):
        displayed = self.viewer.dims.displayed
        if len(displayed) == 2 or self._active_layer is None:
            self.view.visible = False
            return

        self.view.visible = True
        if self.box is not None:
            self.box.parent = None

        extents = self._active_layer.extent.data[:, displayed]
        sizes = extents[1] - extents[0]
        self.box = Box(
            *sizes[::-1],  # swap for vispy
            face_colors=np.repeat([[0, 1, 1]], 12, axis=0),
            edge_color='black',
            parent=self.view.scene,
        )

        self._center = sizes / 2

        self.view.camera.set_range(margin=0.2)

        self._update_depth()
        self._on_camera_change()

    def _update_depth(self):
        # same code as QtViewer._update_camera_depth()
        extent = self.viewer.layers.extent
        extent_all = extent.world[1] - extent.world[0] + extent.step
        extent_displayed = extent_all[list(self.viewer.dims.displayed)]
        diameter = np.linalg.norm(extent_displayed)
        self.view.camera.depth_value = 128 * diameter

    def _on_planes_change(self):
        for i, plane in enumerate(
            self._active_layer.experimental_clipping_planes
        ):
            offset_direction = plane.normal
            offset_position = plane.position
            data_offset = (offset_position - self._center) / offset_direction
            zoom = np.min(
                np.array(self.canvas.size) / self.view.camera.scale_factor
            )
            offset = np.linalg.norm(data_offset) / zoom

            X = self.canvas.size[0]
            x = offset + (X / 2)
            self.lines[i].transform.translate = (x, 0, 0)
            self._offsets[i] = offset

    def _on_camera_change(self):
        if self.box is None:
            return

        # I don't understand why this is needed to align to the napaari view,
        # but it is for some reason :/ TODO figure out why
        side_view = Rotation.from_euler('x', 90, degrees=True)
        mat = np.eye(4)
        mat[:3, :3] = side_view.as_matrix()
        self.box.transform = MatrixTransform(mat)

        # rotate 90 degrees so we see from the side
        up = self.viewer.camera.up_direction
        view = self.viewer.camera.view_direction
        side = Rotation.from_rotvec(np.array(up) * 90, degrees=True).apply(
            view
        )
        self.camera_model.set_view_direction(side, up)

        quat = self.view.camera._quaternion.create_from_euler_angles(
            *self.camera_model.angles,
            degrees=True,
        )
        self.view.camera._quaternion = quat

        VISPY_DEFAULT_ORIENTATION_3D = ('right', 'down', 'away')  # xyz
        self.view.camera.flip = tuple(
            int(ori != default_ori)
            for ori, default_ori in zip(
                self.viewer.camera.orientation[::-1],
                VISPY_DEFAULT_ORIENTATION_3D,
                strict=True,
            )
        )

        self.view.camera.view_changed()
        self.view.camera.center = (self.viewer.camera.center - self._center)[
            ::-1
        ]

        self._update_planes()

    def _on_mouse(self, event=None):
        if self._active_layer is not None and (
            event.type == 'mouse_press' or event.button == 1
        ):
            X = self.canvas.size[0]
            x = event.pos[0]
            offset = x - X / 2
            if offset <= 0:
                self.lines[0].transform.translate = (x, 0, 0)
                self._offsets[0] = offset
            else:
                self.lines[1].transform.translate = (x, 0, 0)
                self._offsets[1] = offset

            self._update_planes()

    def _update_planes(self):
        planes = []
        for offset in self._offsets:
            zoom = np.min(
                np.array(self.canvas.size) / self.view.camera.scale_factor
            )
            data_offset = offset / zoom
            offset_direction = np.array(self.viewer.camera.view_direction)
            offset_position = self._center + offset_direction * data_offset
            planes.append(
                {
                    'position': offset_position,
                    'normal': offset_direction,
                    'enabled': self.planes_enabled.isChecked(),
                }
            )

        # flip normal for second plane
        planes[1]['normal'] *= -1
        with self._active_layer.experimental_clipping_planes.events.blocker(
            self._on_planes_change
        ):
            self._active_layer.experimental_clipping_planes = planes
