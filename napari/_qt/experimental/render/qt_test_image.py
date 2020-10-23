"""QtTestImage class.
"""
import numpy as np
from qtpy.QtWidgets import QFrame, QPushButton, QVBoxLayout

from .qt_labeled_spin_box import LabeledSpinBox
from .test_image import create_tiled_text_array


class QtTestImage(QFrame):
    """Frame with controls to create a new test image.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.

    Attributes
    ----------
    """

    # This is a class attribute to provide a unique index napari-wide.
    image_index = 0

    def __init__(self, viewer, layer):
        super().__init__()
        self.viewer = viewer
        self.layer = layer

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
        name = self._create_image_name()

        return self.viewer.add_image(data, rgb=True, name=name)

    def _create_image_name(self) -> str:
        """Return a name like "test-image-002" for the image.

        Return
        ------
        str
            The image name.
        """
        name = f"test-image-{QtTestImage.image_index:003}"
        QtTestImage.image_index += 1
        return name
