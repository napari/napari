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

        shape = layer.data.shape
        height, width = shape[1:3]  # Which dims are really width/height?

        # Dimension labels.
        self.addWidget(QLabel(f"Shape: {shape}"))
        self.addWidget(QLabel(f"Width: {width}"))
        self.addWidget(QLabel(f"Height: {height}"))


class QtImageInfo(QFrame):
    """Frame showing image shape and dimensions.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()

        layout = QtImageInfoLayout(layer)
        self.setLayout(layout)
