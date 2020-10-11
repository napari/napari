import numpy as np
from vispy.scene.visuals import Line


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation.

    Axes are x=red, y=green, z=blue.
    """

    def __init__(self, viewer, parent=None, order=0):

        self._data_3D = (
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
        self._color_3D = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        self._data_2D = self._data_3D[:4, :2]
        self._color_2D = self._color_3D[:4, :]

        self.viewer = viewer
        self.node = Line(connect='segments', method='gl', parent=parent)
        self.node.order = order

        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        self.viewer.events.axes_visible.connect(self._on_visible_change)

        self._on_ndisplay_change(None)
        self._on_visible_change(None)

    def _on_ndisplay_change(self, event):
        """Change number of displayed axes."""
        if self.viewer.dims.ndisplay == 3:
            self.node.set_data(self._data_3D, color=self._color_3D)
        else:
            self.node.set_data(self._data_2D, color=self._color_2D)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.viewer.axes_visible
