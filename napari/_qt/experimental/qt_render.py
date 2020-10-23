"""QtAsync widget.
"""
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
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


class RenderSpinBox:
    """A SpinBox for the QtRender widget.

    This was cobbled together and is probably not good Qt to emulate,
    current QtRender is a developer visible tool only.
    """

    def __init__(
        self,
        parent,
        label_text: str,
        initial_value: int,
        spin_range: range,
        connect=None,
    ):
        label = QLabel(label_text)
        self.box = self._create_spin_box(initial_value, spin_range)
        self.connect = connect

        if connect is not None:
            self.box.valueChanged.connect(connect)

        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.box)
        parent.addLayout(layout)

    def _create_spin_box(
        self, initial_value: int, spin_range: range
    ) -> QSpinBox:
        """Return one configured QSpinBox.

        Parameters
        ----------
        initial_value : int
            The initial value of the QSpinBox.
        spin_range : range
            The start/stop/step of the QSpinBox.

        Return
        ------
        QSpinBox
            The configured QSpinBox.
        """
        box = QSpinBox()
        box.setKeyboardTracking(False)
        box.setMinimum(spin_range.start)
        box.setMaximum(spin_range.stop)
        box.setSingleStep(spin_range.step)
        box.setAlignment(Qt.AlignCenter)
        box.setValue(initial_value)
        box.valueChanged.connect(self._on_change)
        return box

    def _on_change(self, value) -> None:
        """Called when the spin box value was changed."""
        # We must clearFocus or it would double-step, no idea why.
        self.box.clearFocus()

        # Notify any connection we have.
        if self.connect is not None:
            self.connect(value)

    def value(self) -> int:
        """Return the current value of the QSpinBox."""
        return self.box.value()

    def set(self, value) -> None:
        """Set the current value of the QSpinBox."""
        self.box.setValue(value)


class QtImageInfo(QFrame):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.setLayout(self._create_layout())

        # Get initial value and hook to event.
        self._on_octree_level()
        self.layer.events.octree_level.connect(self._on_octree_level)

    def _create_layout(self) -> QLayout:
        """Create layout for image info."""
        layout = QVBoxLayout()
        max_level = self.layer.num_octree_levels - 1
        self.spin_level = RenderSpinBox(
            layout,
            "Octree Level",
            max_level,
            range(0, max_level, 1),
            connect=self._on_new_level,
        )
        self._add_dimensions(layout)
        return layout

    def _add_dimensions(self, layout: QLayout) -> None:
        """Add dimension labels to layout.

        Parameters
        ----------
        layout : QLayout
            Add dimension labels to this layout.
        """
        height, width = self.layer.data.shape[1:3]  # fix dims
        layout.addWidget(QLabel(f"Image Width: {width}"))
        layout.addWidget(QLabel(f"Image Height: {height}"))

    def _on_new_level(self, value):
        """Level spinbox changed.

        Parameters
        ----------
        value : int
            New value of the spinbox
        """
        self.layer.octree_level = value

    def _on_octree_level(self, event=None):
        """Set SpinBox to match the layer's new octree_level."""
        value = self.layer.octree_level
        self.spin_level.set(value)


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

        layout = QVBoxLayout()
        layout.addStretch(1)
        size_range = range(1, 2048, 100)
        self.width = RenderSpinBox(layout, "Image Width", 1024, size_range)
        self.height = RenderSpinBox(layout, "Image Height", 1024, size_range)

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
