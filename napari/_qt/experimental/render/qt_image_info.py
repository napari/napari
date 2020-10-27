"""QtImageInfo class.
"""


from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout


class QtImageInfoLayout(QVBoxLayout):
    """Layout of the image info frame.

    Parameters
    ----------
    layer : Layer
        Show info for this layer.
    """

    def __init__(self, layer):
        super().__init__()

        # TODO_OCTREE: This class was going to have more, if not
        # we can get rid of it.
        shape = layer.data.shape
        self.addWidget(QLabel(f"Shape: {shape}"))


class QtImageInfo(QFrame):
    """Frame showing image shape and dimensions.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()

        layout = QtImageInfoLayout(layer)
        self.setLayout(layout)
