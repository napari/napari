"""QtImageInfo class.
"""


from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout


class QtImageInfoLayout(QVBoxLayout):
    """Image info is generic to all image layers (not just Octree).

    Parameters
    ----------
    layer : Layer
        Show info for this layer.
    """

    def __init__(self, layer):
        super().__init__()

        # TODO_OCTREE: This class was going to have more, if not
        # we can role this into QtRender itself?
        shape = layer.data.shape
        self.addWidget(QLabel(f"Shape: {shape}"))


class QtImageInfo(QFrame):
    """Frame showing image shape.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()

        layout = QtImageInfoLayout(layer)
        self.setLayout(layout)
