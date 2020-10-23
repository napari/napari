"""QtTestImage class.
"""
from typing import Callable, Tuple

import numpy as np
from qtpy.QtWidgets import QFrame, QPushButton, QVBoxLayout

from .qt_labeled_spin_box import LabeledSpinBox
from .test_image import create_tiled_text_array


class QtTestImageLayout(QVBoxLayout):
    """Layout for the Test Image frame.

    Parameters
    ----------
    on_create : Callable[[], None]
        Called when the create test image button is pressed.
    """

    def __init__(self, on_create: Callable[[], None]):
        super().__init__()
        self.addStretch(1)

        # Dimension controls.
        size_range = range(1, 65536, 100)
        self.width = LabeledSpinBox(self, "Image Width", 1024, size_range)
        self.height = LabeledSpinBox(self, "Image Height", 1024, size_range)

        # Create test image button.
        button = QPushButton("Create Test Image")
        button.setToolTip("Create a new test image")
        button.clicked.connect(on_create)
        self.addWidget(button)

    def get_size(self) -> Tuple[int, int]:
        return (self.width.value(), self.height.value())


class QtTestImage(QFrame):
    """Frame with controls to create a new test image.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    layer : Layer
        The layer we are hook up to.
    """

    # This is a class attribute so that we use a unique index napari-wide,
    # not just within in this one QtRender widget, this one layer.
    image_index = 0

    def __init__(self, viewer, layer):
        super().__init__()
        self.viewer = viewer
        self.layer = layer
        self.layout = QtTestImageLayout(self._create_test_image)
        self.setLayout(self.layout)

    def _create_test_image(self) -> None:
        """Create a new test image."""
        size = self.layout.get_size()
        images = [create_tiled_text_array(x, 16, 16, size) for x in range(5)]
        data = np.stack(images, axis=0)

        unique_name = f"test-image-{QtTestImage.image_index:003}"
        QtTestImage.image_index += 1

        self.viewer.add_image(data, rgb=True, name=unique_name)
