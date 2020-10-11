import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.visuals.transforms import STTransform


class VispyScaleBarVisual:
    """Scale bar in world coordinates.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._data = np.array(
            [
                [0, 0, -1],
                [1, 0, -1],
                [0, -5, -1],
                [0, 5, -1],
                [1, -5, -1],
                [1, 5, -1],
            ]
        )
        self._color = [1, 1, 1, 1]
        self._target_length = 100
        self.viewer = viewer
        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.order = order
        self.node.set_data(self._data, color=self._color)
        self.node.transform = STTransform()
        self.node.transform.translate = [66, 14, 0, 0]

        self.text_node = Text(pos=[0, 0], parent=parent)
        self.text_node.order = order
        self.text_node.color = self._color
        self.text_node.transform = STTransform()
        self.text_node.transform.translate = [33, 16, 0, 0]
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        self.viewer.events.scale_bar_visible.connect(self._on_visible_change)
        self._on_visible_change(None)

        self._scale = 0.1
        self.update_scale(1)

    def update_scale(self, scale):
        """Update scale bar length based on canvas2world scale.

        Parameters
        ----------
        scale : float
            Scale going from canvas pixels to world coorindates.
        """
        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            print('ee')
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        target_world_pixels = scale_canvas2world * target_canvas_pixels

        # Round scalebar to nearest factor of 1, 2, 5, 10 in world pixels on a log scale
        resolutions = [1, 2, 5, 10]
        log_target = np.log10(target_world_pixels)
        mult_ind = np.argmin(
            np.abs(np.subtract(np.log10(resolutions), log_target % 1))
        )
        power_val = np.floor(log_target)
        target_world_pixels_rounded = resolutions[mult_ind] * np.power(
            10, power_val
        )

        # Convert back to canvas pixels to get actual length of scale bar
        target_canvas_pixels_rounded = (
            target_world_pixels_rounded / scale_canvas2world
        )
        scale = target_canvas_pixels_rounded

        # Update scalebar and text
        self.node.transform.scale = [scale, 1, 1, 1]
        self.text_node.text = f'{target_world_pixels_rounded:.4g}'

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.viewer.scale_bar_visible
        self.text_node.visible = self.viewer.scale_bar_visible
