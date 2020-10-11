import numpy as np
from vispy.scene.visuals import Line, Text
from vispy.visuals.transforms import STTransform


class VispyWelcomeVisual:
    """Welcome to napari visual.
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
        self._target_length = 100
        self.viewer = viewer
        self.node = Line(
            connect='segments', method='gl', parent=parent, width=3
        )
        self.node.order = order
        self.node.set_data(self._data, [0.7, 0.7, 0.7, 1])
        self.node.transform = STTransform()
        self.node.transform.translate = [66, 14, 0, 0]

        self.text_node = Text(pos=[0, 0], parent=parent)
        self.text_node.order = order
        self.text_node.transform = STTransform()
        self.text_node.transform.translate = [300, 400, 0, 0]
        self.text_node.font_size = 30
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = 'Drag file(s) here to open or use File Open'
        self.text_node.color = [0.7, 0.7, 0.7, 1]

        self.viewer.events.layers_change.connect(self._on_visible_change)
        self._on_visible_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        visible = len(self.viewer.layers) == 0
        self.node.visible = visible
        self.text_node.visible = visible
