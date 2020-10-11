import numpy as np
from vispy.scene.visuals import Line


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation.

    Axes are x=red, y=green, z=blue.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._data = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ]
            )
            * 100
        )
        self._color = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        self.viewer = viewer
        self.node = Line(connect='segments', method='gl', parent=parent)
        self.node.order = order
        self.node.set_data(self._data, color=self._color)

        self.viewer.events.axes_visible.connect(self._on_visible_change)
        self._on_visible_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.viewer.axes_visible
