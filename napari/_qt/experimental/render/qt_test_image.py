"""QtTestImage class.
"""
import numpy as np
from qtpy.QtWidgets import QFrame, QPushButton, QVBoxLayout

from .qt_labeled_spin_box import LabeledSpinBox
from .test_image import create_tiled_text_array


class ImageNamer:
    """Provides unique name for test images.

    The index is global, so it does not matter which QtRender widget,
     which layer, is creating the image, the next available name
    is always given.
    """

    image_index = 0

    def get_name(self) -> str:
        """Return a name like "test-image-002" for the image.

        Return
        ------
        str
            The image name.
        """
        index = self.image_index
        self.image_index += 1
        return f"test-image-{index:003}"


class QtTestImage(QFrame):
    """Frame with controls to create a new test image.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.

    Attributes
    ----------
    """

    def __init__(self, viewer, layer):
        super().__init__()
        self.viewer = viewer
        self.layer = layer
        self.namer = ImageNamer()

        layout = QVBoxLayout()
        layout.addStretch(1)
        size_range = range(1, 2048, 100)
        self.width = LabeledSpinBox(layout, "Image Width", 1024, size_range)
        self.height = LabeledSpinBox(layout, "Image Height", 1024, size_range)

        create = QPushButton("Create Test Image")
        create.setToolTip("Create a new test image")
        create.clicked.connect(self._create_test_image)

        layout.addWidget(create)
        self.setLayout(layout)
        self.image_index = 0

    def _create_test_image(self):
        """Create a new test image."""
        size = (self.width.value(), self.height.value())
        images = [create_tiled_text_array(x, 16, 16, size) for x in range(5)]
        data = np.stack(images, axis=0)
        name = self.namer.get_name()

        return self.viewer.add_image(data, rgb=True, name=name)
