import numpy as np
from vispy.scene.visuals import Line
from vispy.visuals.transforms import STTransform


def make_dashed_line(num_dashes, axis):
    """Make a dashed line.

    Parameters
    ----------
    num_dashes : int
        Numbers of dashes in the line.
    axis :
        Axis which is dashed.

    Returns
    -------
    np.ndarray
        Dashed line.
    """
    dashes = np.linspace(0, 1, num_dashes * 2)
    dashed_line_ends = np.concatenate(
        [[dashes[2 * i], dashes[2 * i + 1]] for i in range(num_dashes)], axis=0
    )
    dashed_line = np.zeros((2 * num_dashes, 3))
    dashed_line[:, axis] = np.array(dashed_line_ends)
    return dashed_line


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation."""

    def __init__(self, axes, parent=None, order=0):
        self.axes = axes

        self._default_data = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )
        self._default_color = np.array(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 1],
            ]
        )
        self._dashed_data = np.concatenate(
            [
                [[0, 0, 0], [1, 0, 0]],
                make_dashed_line(4, axis=1),
                make_dashed_line(8, axis=2),
            ],
            axis=0,
        )
        self._dashed_color = np.concatenate(
            [
                [[0, 1, 1, 1], [0, 1, 1, 1]],
                [[1, 1, 0, 1]] * 4 * 2,
                [[1, 0, 1, 1]] * 8 * 2,
            ],
            axis=0,
        )

        self._target_length = 100
        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.transform = STTransform()
        self.node.order = order

        self.axes.events.visible.connect(self._on_visible_change)
        self.axes.events.colored.connect(self._on_data_change)
        self.axes.events.dashed.connect(self._on_data_change)

        self._on_visible_change(None)
        self._on_data_change(None)

        self._scale = 0.1
        self.update_scale(1)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.axes.visible

    def _on_data_change(self, event):
        """Change style of axes."""
        if self.axes.dashed:
            data = self._dashed_data
        else:
            data = self._default_data

        if not self.axes.colored:
            color = np.subtract(1, self.axes.background_color)[:3]
        else:
            if self.axes.dashed:
                color = self._dashed_color
            else:
                color = self._default_color
        self.node.set_data(data, color)

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
