"""QtAsync widget.
"""
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .test_image import create_tiled_text_array

# Global index for image names.
image_index = 0


def _get_image_name() -> str:
    """Return a name like "test-image-002" for the image.

    Use a global so not matter which QtRender widget you use, you get a
    unique name.
    """
    global image_index
    index = image_index
    image_index += 1
    return f"test-image-{index:003}"


# Global so no matter where you create the test image it increases.
test_image_index = 0


class QtCreateTestImageButton(QPushButton):
    """Push button to create a new test image.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    """

    def __init__(self, viewer, slot=None):
        super().__init__("Create Test Image")

        self.viewer = viewer
        self.setToolTip("Add a new test image.")

        if slot is not None:
            self.clicked.connect(slot)


class SpinBox:
    def __init__(
        self, parent, label_text: str, initial_value: int, spin_range: range
    ):
        label = QLabel(label_text)

        box = QSpinBox()
        box.setKeyboardTracking(False)
        box.setMinimum(spin_range.start)
        box.setMaximum(spin_range.stop)
        box.setSingleStep(spin_range.step)
        box.setAlignment(Qt.AlignCenter)
        box.setValue(initial_value)
        self.box = box

        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(box)
        parent.addLayout(layout)

    def value(self) -> int:
        return self.box.value()


class QtTestImage(QFrame):
    """Frame with controls to create a new test image.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.

    Attributes
    ----------
    """

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        layout = QVBoxLayout()
        layout.addStretch(1)
        size_range = range(1, 2048, 100)
        self.width = SpinBox(layout, "Image Width", 1024, size_range)
        self.height = SpinBox(layout, "Image Height", 1024, size_range)

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
        return self.viewer.add_image(data, rgb=True, name=_get_image_name())


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

        self.layer.events.octree_level.connect(self._on_octree_level)
        layout = QVBoxLayout()

        spin_layout = QHBoxLayout()

        self.spin_level = QSpinBox()
        self.spin_level.setKeyboardTracking(False)
        self.spin_level.setSingleStep(1)
        self.spin_level.setMinimum(0)
        self.spin_level.setMaximum(10)
        self.spin_level.valueChanged.connect(self._on_spin)
        self.spin_level.setAlignment(Qt.AlignCenter)

        label = QLabel("Octree Level:")
        spin_layout.addWidget(label)
        spin_layout.addWidget(self.spin_level)

        layout.addLayout(spin_layout)
        layout.addStretch(1)
        layout.addWidget(QtTestImage(viewer))
        self.setLayout(layout)

        # Get initial value.
        self._on_octree_level()

    def _on_spin(self, value):
        """Level spinbox changed.

        Parameters
        ----------
        value : int
            New value of the spinbox
        """
        self.layer.octree_level = value

        # Focus stuff to prevent double-stepping.
        self.spin_level.clearFocus()
        self.setFocus()

    def _on_octree_level(self, event=None):
        """Set SpinBox to match the layer's new octree_level."""
        value = self.layer.octree_level
        self.spin_level.setValue(value)
