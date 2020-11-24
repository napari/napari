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
        # TODO_OCTREE: The OctreeLevelInfo and SliceConfig and
        # their attributes are messy. This becomes clear trying
        # to send a coherent message to the monitor. Ideally
        # the message comes from the octree code directly, and
        # references the data it stores (names, etc).
        intersection = self.layer.get_intersection()
        level = intersection.level
        image_shape = level.info.image_shape
        shape_in_tiles = level.info.shape_in_tiles

        # slice_config are properties of the whole octree, not
        # specific to this level. Confusingly.
        slice_config = level.info.slice_config
        base_shape = slice_config.base_shape
        tile_size = slice_config.tile_size

        data = {
            "tile_config": {
                "base_shape": base_shape,
                "image_shape": image_shape,
                "shape_in_tiles": shape_in_tiles,
                "tile_size": tile_size,
            }
        }

        monitor.add(data)

        # Create seen, a list of (row, col) pairs of tiles which were
        # visible in the intersection. When the octree code provides
        # this message, it can do it more efficiently than querying
        # every possible pair!
        shape = shape_in_tiles
        seen = []
        for row in range(shape[0]):
            for col in range(shape[1]):
                if intersection.is_visible(row, col):
                    seen.append([row, col])

        # If we create our own json encoder for numpy (like vispy does)
        # then we don't need to call tolist() here.
        monitor.add(
            {
                "tile_state": {
                    "seen": seen,
                    "corners": intersection.corners.tolist(),
                }
            }
        )

    def _on_camera_move(self, _event=None):
        """Called when the camera was moved."""
        if SHOW_MINIMAP:
            self.mini_map.update()
        self.frame_rate.on_camera_move()

        if monitor:
            self._monitor()
