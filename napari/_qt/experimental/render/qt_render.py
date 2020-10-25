"""QtRender widget.
"""


from qtpy.QtWidgets import QVBoxLayout, QWidget

from ....layers.image import Image
from ....layers.image.experimental.octree_image import OctreeImage
from .qt_image_info import QtImageInfo, QtOctreeInfo
from .qt_mini_map import MiniMap
from .qt_test_image import QtTestImage


class QtRender(QWidget):
    """Dockable widget for render controls.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    layer : Optional[Layer]
        Show controls for this layer, or test image controls if no layer.
    """

    def __init__(self, viewer, layer=None):
        """Create our windgets.
        """
        super().__init__()
        self.viewer = viewer
        self.layer = layer

        layout = QVBoxLayout()

        if isinstance(layer, Image):
            layout.addWidget(QtImageInfo(layer))

        if isinstance(layer, OctreeImage):
            layout.addWidget(QtOctreeInfo(layer))

            self.mini_map = MiniMap(layer)
            layout.addWidget(self.mini_map)
            self.viewer.camera.events.center.connect(self._update)

        layout.addStretch(1)
        layout.addWidget(QtTestImage(viewer))
        self.setLayout(layout)

    def _update(self, event=None):
        self.mini_map.update()
