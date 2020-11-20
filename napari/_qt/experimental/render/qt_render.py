"""QtRender widget.
"""
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ....components.experimental import monitor
from ....layers.image import Image
from ....layers.image.experimental.octree_image import OctreeImage
from .qt_frame_rate import QtFrameRate
from .qt_image_info import QtImageInfo
from .qt_mini_map import QtMiniMap
from .qt_octree_info import QtOctreeInfo
from .qt_test_image import QtTestImage

SHOW_MINIMAP = False


class QtRender(QWidget):
    """Dockable widget for render controls.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    layer : Optional[Layer]
        Show controls for this layer. If no layer show minimal controls.
    """

    def __init__(self, viewer, layer=None):
        """Create our windgets.
        """
        super().__init__()
        self.viewer = viewer
        self.layer = layer

        layout = QVBoxLayout()

        # Basic info for any image layer.
        if isinstance(layer, Image):
            layout.addWidget(QtImageInfo(layer))

        if isinstance(layer, OctreeImage):
            # Octree specific controls and widgets.
            layout.addWidget(QtOctreeInfo(layer))

            if SHOW_MINIMAP:
                self.mini_map = QtMiniMap(layer)
                layout.addWidget(self.mini_map)

            self.viewer.camera.events.center.connect(self._on_camera_move)

        # Controls to create a new test image.
        layout.addStretch(1)
        layout.addWidget(QtTestImage(viewer))

        # Frame rate meter.
        self.frame_rate = QtFrameRate()
        layout.addWidget(self.frame_rate)

        self.setLayout(layout)

    def _monitor(self):
        intersection = self.layer.get_intersection()
        level = intersection.level
        shape = level.info.shape_in_tiles

        monitor.add({"tile_config": {"rows": shape[0], "cols": shape[1]}})

        seen = []
        for row in range(shape[0]):
            for col in range(shape[1]):
                if intersection.is_visible(row, col):
                    seen.append([row, col])
        monitor.add({"tile_state": {"seen": seen}})

        # TODO_MON: Move this to somewhere central!
        monitor.poll()

        # TODO_MON: let users of the monitor register events, and get
        # notified using those events? For check for now.
        if monitor.service is not None:
            data = monitor.service.from_client
            try:
                self.layer.show_grid = data['show_grid']
            except KeyError:
                pass  # no client setting, that's fine

    def _on_camera_move(self, _event=None):
        """Called when the camera was moved."""
        if SHOW_MINIMAP:
            self.mini_map.update()
        self.frame_rate.on_camera_move()
        self._monitor()
