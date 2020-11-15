"""QtImageInfo class.
"""
from typing import List

from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout


def _get_shape(data):
    if isinstance(data, List):
        return data[0].shape
    return data.shape


class QtImageInfoLayout(QVBoxLayout):
    """Image info is generic to all image layers (not just Octree).

    Parameters
    ----------
    layer : Layer
        Show info for this layer.
    """

    # TODO_OCTREE: This class was going to have more, if not
    # we can role this into QtRender itself?

    def __init__(self, layer):
        super().__init__()

        shape = _get_shape(layer.data)
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
