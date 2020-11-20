"""QtRender widget.
"""
from qtpy.QtCore import QTimer
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

        # We have no layer sometimes because QtRender
        # is shown even when there is no layer present.
        if self.layer is not None:
            self._timer = QTimer()
            self._timer.setInterval(33)
            self._timer.timeout.connect(self._on_timer)
            self._timer.start()

    def _on_timer(self) -> None:

        # To send out data we could poll only when we've updated something.
        # Right now to receive data we have to constantly poll. Not sure
        # how to avoid polling but maybe there is a way?
        monitor.poll()

        # TODO_MON: let users of the monitor register events, and get
        # notified using those events? For check for now.
        if monitor.service is not None:
            data = monitor.service.from_client
            try:
                show = data['show_grid']
                print(f"Monitor: setting grid to {show}")
                self.layer.show_grid = show
            except KeyError:
                pass  # no client setting, that's fine

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

        # Zooming starves our timer, so poll here for now. Utlimately
        # polling for input should go away totally.
        monitor.poll()

    def _on_camera_move(self, _event=None):
        """Called when the camera was moved."""
        if SHOW_MINIMAP:
            self.mini_map.update()
        self.frame_rate.on_camera_move()
        self._monitor()
