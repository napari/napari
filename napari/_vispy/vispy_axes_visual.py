import numpy as np
from vispy.scene.visuals import Line
from vispy.visuals.transforms import STTransform


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation.

    Axes are x=red, y=green, z=blue.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._data = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )
        self._color = np.array(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 1],
            ]
        )
        self._target_length = 100
        self.viewer = viewer
        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.transform = STTransform()
        self.node.order = order
        self.node.set_data(self._data, color=self._color)

        self.viewer.events.axes_visible.connect(self._on_visible_change)
        self._on_visible_change(None)

        self._scale = 0.1
        self.update_scale(1)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.viewer.axes_visible

    def update_scale(self, scale):
        """Update axes length based on canvas2world scale.

        Parameters
        ----------
        scale : float
            Scale going from canvas pixels to world coorindates.
        """
        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        scale = target_canvas_pixels * scale_canvas2world

        # Update axes scale
        self.node.transform.scale = [scale, scale, scale, 1]
