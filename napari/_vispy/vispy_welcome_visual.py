from os.path import dirname, join

import numpy as np
from imageio import imread
from vispy.scene.visuals import Text
from vispy.visuals.transforms import STTransform

from ..utils.misc import str_to_rgb
from .image import Image as ImageNode


class VispyWelcomeVisual:
    """Welcome to napari visual.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._viewer = viewer

        # Load logo and make grayscale
        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        logo = imread(logopath)
        self._raw_logo = logo

        new_logo = np.zeros(logo.shape)
        logo_border = np.all(logo[..., :3] == [38, 40, 61], axis=2)
        new_logo[logo_border, :3] = np.divide(
            str_to_rgb(self._viewer.palette['foreground']), 255
        )
        new_logo[np.invert(logo_border), :3] = np.divide(
            str_to_rgb(self._viewer.palette['background']), 255
        )
        new_logo[..., -1] = logo[..., -1] * 0.9
        # print(logo_gray.shape)
        # print(logo_gray.max())
        # #logo_gray = np.mean(logo[..., :3], axis=2)
        # logo[..., :3] = np.stack([logo_gray, logo_gray, logo_gray], axis=2)
        # logo[..., -1] = logo[..., -1] * 0.9

        self._logo = new_logo
        self.node = ImageNode(parent=parent)
        self.node.order = order

        self.node.set_data(self._logo)
        self.node.cmap = 'gray'
        self.node.transform = STTransform()

        self.text_node = Text(pos=[0, 0], parent=parent, method='gpu')
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = (
            'To get started\n'
            'drag and drop file(s) here,\n'
            'use File : Open File(s),\n'
            'or call a viewer.add_* method!'
        )
        self.text_node.color = [1, 1, 1, 0.6]

        self._viewer.events.layers_change.connect(self._on_visible_change)
        self._on_visible_change(None)
        self._on_canvas_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        visible = len(self._viewer.layers) == 0
        self.node.visible = visible
        self.text_node.visible = visible

    def _on_canvas_change(self, event):
        """Change visibiliy of axes."""
        if self.node.canvas is not None:
            center = np.divide(self.node.canvas.size, 2)
        else:
            center = np.array([256, 256])

        # Calculate some good default positions for the logo and text
        center_logo = [
            center[0] - center[1] / 2.4,
            1 / 2 * center[1] - center[1] / 2.4,
        ]
        self.node.transform.translate = [center_logo[0], center_logo[1], 0, 0]
        self.node.transform.scale = [
            center[1] / 1.2 / self._logo.shape[0],
            center[1] / 1.2 / self._logo.shape[0],
            0,
            0,
        ]

        self.text_node.font_size = center[1] / 15
        self.text_node.transform.translate = [
            center[0],
            4 / 3 * center[1],
            0,
            0,
        ]
