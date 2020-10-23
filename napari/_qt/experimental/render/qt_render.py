"""QtAsync widget.
"""
import numpy as np
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from .qt_image_info import QtImageInfo
from .qt_test_image import QtTestImage

# Global so no matter where you create the test image it increases.
test_image_index = 0


class QtRender(QWidget):
    """Dockable widget for render controls.

    Attributes
    ----------
    """

    def __init__(self, viewer, layer):
        """Create our windgets.
        """
        super().__init__()
        self.layer = layer

        layout = QVBoxLayout()

        layout.addWidget(QtImageInfo(layer))

        self.mini_map = QLabel()
        layout.addWidget(self.mini_map)

        layout.addStretch(1)
        layout.addWidget(QtTestImage(viewer, layer))
        self.setLayout(layout)

        data = np.zeros((50, 50, 4), dtype=np.uint8)
        data[:, 25, :] = (255, 255, 255, 255)
        data[25, :, :] = (255, 255, 255, 255)

        image = QImage(
            data, data.shape[1], data.shape[0], QImage.Format_RGBA8888,
        )
        self.mini_map.setPixmap(QPixmap.fromImage(image))
