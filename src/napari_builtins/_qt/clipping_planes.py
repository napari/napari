import numpy as np
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from scipy.spatial.transform import Rotation
from superqt import QToggleSwitch
from vispy.scene import ArcballCamera, Box, InfiniteLine, SceneCanvas
from vispy.visuals.transforms import MatrixTransform, STTransform

import napari

# TODO: scaled layer mess up the math :/
# TODO: allow adding multiple and controlling direction
# TODO: allow "locking" planes, hiding them from view and leaving them there
#       (currently doable by deselecting all layers)


class ClippingPlanesControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
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
        self.planes_enabled.setChecked(True)

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
        self.viewer.layers.selection.events.connect(self._on_extent_change)
        self.canvas.events.mouse_move.connect(self._on_mouse)
        self.canvas.events.mouse_press.connect(self._on_mouse)
        self.planes_enabled.toggled.connect(self._update_planes)

        self._on_extent_change()

    def _on_extent_change(self):
        displayed = self.viewer.dims.displayed
        if len(displayed) == 2 or not self.viewer.layers.selection:
            self.view.visible = False
            return

        self.view.visible = True
        if self.box is not None:
            self.box.parent = None

        extents = self.viewer.layers.get_extent(
            self.viewer.layers.selection
        ).world[:, displayed]
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
        if self.viewer.layers.selection and (
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
        planes_world = []
        for offset in self._offsets:
            zoom = np.min(
                np.array(self.canvas.size) / self.view.camera.scale_factor
            )
            data_offset = offset / zoom
            offset_direction = np.array(self.viewer.camera.view_direction)
            offset_position = self._center + offset_direction * data_offset
            planes_world.append([offset_position, offset_direction])

        # flip normal for second plane
        planes_world[1][1] *= -1

        # convert to data coordinates
        displayed = list(self.viewer.dims.displayed)
        for layer in self.viewer.layers.selection:
            if layer.ndim < 3:
                continue

            planes_data = []
            for plane in planes_world:
                world_pos_full = np.zeros(self.viewer.dims.ndim)
                world_pos_full[displayed] = plane[0]
                data_pos = layer.world_to_data(world_pos_full)[displayed]

                world_norm_full = np.zeros(self.viewer.dims.ndim)
                world_norm_full[displayed] = plane[1]
                data_norm = layer.world_to_data(world_norm_full)[displayed]

                planes_data.append(
                    {
                        'position': data_pos,
                        'normal': data_norm,
                        'enabled': self.planes_enabled.isChecked(),
                    }
                )
            layer.experimental_clipping_planes = planes_data
