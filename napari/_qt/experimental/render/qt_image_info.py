"""QtImageInfo class.
"""
from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout


def _get_shape(data) -> tuple:
    """Get shape of the data.

    Return
    ------
    tuple
        The shape of the data.
    """
    if isinstance(data, list):
        return data[0].shape  # Level zero of multiscale.
    return data.shape  # Single scale.


class QtImageInfo(QFrame):
    """Frame showing image shape.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()
        layout = QVBoxLayout()
        shape = _get_shape(layer.data)
        layout.addWidget(QLabel(f"Shape: {shape}"))
        self.setLayout(layout)
