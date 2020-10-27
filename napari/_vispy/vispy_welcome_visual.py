from os.path import dirname, join

import numpy as np
import scipy.ndimage as ndi
from imageio import imread
from vispy.scene.visuals import Text
from vispy.visuals.transforms import STTransform

from ..utils.misc import str_to_rgb
from ..utils.theme import darken, lighten
from .image import Image as ImageNode


class VispyWelcomeVisual:
    """Welcome to napari visual.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._viewer = viewer

        # Load logo and make grayscale
        logopath = join(dirname(__file__), '..', 'resources', 'logo.png')
        logo = imread(logopath)
        self._logo_raw = logo
        self._logo_border = np.all(logo[..., :3] == [38, 40, 61], axis=2)
        self._logo = np.zeros(logo.shape)

        self.node = ImageNode(parent=parent)
        self.node.order = order

        self.node.cmap = 'grays'
        self.node.transform = STTransform()

        self.text_node = Text(
            pos=[0, 0], parent=parent, method='gpu', bold=False
        )
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.anchors = ('left', 'center')
        self.text_node.text = (
            'to add data:\n'
            '   - drag and drop file(s) here\n'
            '   - select File > Open from the menu\n'
            '   - call a viewer.add_* method'
        )
        self.text_node.color = np.divide(
            str_to_rgb(darken(self._viewer.palette['foreground'], 30)), 255
        )

        self._on_palette_change(None)
        self._on_visible_change(None)
        self._on_canvas_change(None)

    def _on_palette_change(self, event):
        """Change colors of the logo and text."""
        if (
            np.mean(str_to_rgb(self._viewer.palette['background'])[:3])
            < 255 / 2
        ):
            foreground_color = np.divide(
                str_to_rgb(darken(self._viewer.palette['foreground'], 50)), 255
            )
            background_color = np.divide(
                str_to_rgb(darken(self._viewer.palette['background'], 70)), 255
            )
            # Note this unsual scaling is done to preserve color balance on
            # rendering by VisPy, which appears to be off when opacity < 1
            text_color = np.multiply(foreground_color, [0.4, 0.65, 0.9])
        else:
            foreground_color = np.divide(
                str_to_rgb(lighten(self._viewer.palette['foreground'], 30)),
                255,
            )
            background_color = np.divide(
                str_to_rgb(lighten(self._viewer.palette['background'], 30)),
                255,
            )
            text_color = np.divide(
                str_to_rgb(darken(self._viewer.palette['background'], 60)), 255
            )

        new_logo = np.zeros(self._logo_raw.shape)
        new_logo[self._logo_border, :3] = foreground_color
        new_logo[np.invert(self._logo_border), :3] = background_color
        new_logo[..., -1] = self._logo_raw[..., -1] * 0.7

        # Do a convolution to smooth any pixelation
        kernel = np.array([[0, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0]])
        kernel = np.expand_dims(kernel / np.sum(kernel), axis=2)
        new_logo = ndi.convolve(new_logo, kernel)

        self._logo = new_logo
        self.node.set_data(self._logo)
        # Having opacity < 1 improves blending but throws color balance
        # off which needs to be adjusted if desired
        self.text_node.color = list(text_color) + [0.7]

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
            2 / 3 * center[1] - center[1] / 3,
        ]
        self.node.transform.translate = [center_logo[0], center_logo[1], 0, 0]
        self.node.transform.scale = [
            center[1] / 1.2 / self._logo.shape[0],
            center[1] / 1.2 / self._logo.shape[0],
            0,
            0,
        ]

        self.text_node.font_size = center[1] / 24
        self.text_node.transform.translate = [
            center[0] - center[1] / 2.4,
            1.45 * center[1],
            0,
            0,
        ]
