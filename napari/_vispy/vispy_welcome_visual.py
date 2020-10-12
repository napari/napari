from os.path import dirname, join

import numpy as np
from imageio import imread
from vispy.scene.visuals import Text
from vispy.visuals.transforms import STTransform

from .image import Image as ImageNode


class VispyWelcomeVisual:
    """Welcome to napari visual.
    """

    def __init__(self, viewer, parent=None, order=0):

        if parent is not None:
            center = np.divide(parent.canvas.size, 2)
        else:
            center = np.array([256, 256])

        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        logo = imread(logopath)
        logo_colored = np.mean(logo[..., :3], axis=2)
        logo[..., :3] = np.stack(
            [logo_colored, logo_colored, logo_colored], axis=2
        )
        self._logo = logo
        self.viewer = viewer
        self.node = ImageNode(parent=parent)
        self.node.order = order

        self.node.set_data(self._logo)
        self.node.cmap = 'gray'
        self.node.transform = STTransform()
        center_logo = [center[0] - 100, 1 / 2 * center[1] - 100]
        self.node.transform.translate = [center_logo[0], center_logo[1], 0, 0]
        self.node.transform.scale = [
            200 / self._logo.shape[0],
            200 / self._logo.shape[0],
            0,
            0,
        ]

        self.text_node = Text(pos=[0, 0], parent=parent)
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.transform.translate = [
            center[0],
            4 / 3 * center[1],
            0,
            0,
        ]
        self.text_node.font_size = 25
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = (
            'Drag file(s) here to open \n or \n use File -> Open File(s)'
        )
        self.text_node.color = [0.7, 0.7, 0.7, 1]

        self.viewer.events.layers_change.connect(self._on_visible_change)
        self._on_visible_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        visible = len(self.viewer.layers) == 0
        self.node.visible = visible
        self.text_node.visible = visible
