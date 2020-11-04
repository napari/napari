"""QtTestImage and QtTestImageLayout classes.
"""
from collections import namedtuple
from typing import Callable, Tuple

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)
from skimage import data as skimage_data

from ....utils import config
from .qt_labeled_spin_box import QtLabeledSpinBox
from .test_image import create_test_image

Callback = Callable[[], None]
IntCallback = Callable[[int], None]

TILE_SIZE_DEFAULT = 64
TILE_SIZE_RANGE = range(1, 4096, 100)

IMAGE_SHAPE_DEFAULT = (1024, 1024)  # (height, width)
IMAGE_SHAPE_RANGE = range(1, 65536, 100)

# The test images which QtTestImage can create. There is a drop-down and
# the user can pick any one of these. Some are fixed shape, while others
# allow you to request a specific shape.
TEST_IMAGES = {
    "Digits": {
        "shape": None,
        "factory": lambda image_shape: create_test_image(
            "0", (16, 16), image_shape
        ),
    },
    "Astronaut": {
        "shape": (512, 512),
        "factory": lambda: skimage_data.astronaut(),
    },
    "Chelsea": {
        "shape": (300, 451),
        "factory": lambda: skimage_data.chelsea(),
    },
    "Coffee": {"shape": (400, 600), "factory": lambda: skimage_data.coffee()},
}


class QtSetShape(QGroupBox):
    """Controls to set the shape of an image."""

    def __init__(self):
        super().__init__("Dimensions")

        layout = QVBoxLayout()
        self.height = QtLabeledSpinBox(
            "Height", IMAGE_SHAPE_DEFAULT[0], IMAGE_SHAPE_RANGE
        )
        layout.addLayout(self.height)
        self.width = QtLabeledSpinBox(
            "Width", IMAGE_SHAPE_DEFAULT[1], IMAGE_SHAPE_RANGE
        )
        layout.addLayout(self.width)
        self.setLayout(layout)

    def get_shape(self) -> Tuple[int, int]:
        """Return the currently configured shape.

        Return
        ------
        Tuple[int, int]
            The requestsed [height, width] shape.
        """
        return self.height.spin.value(), self.width.spin.value()


class QtFixedShape(QGroupBox):
    """Controls to display the fixed shape of an image."""

    def __init__(self):
        super().__init__("Details")

        layout = QVBoxLayout()
        self.shape = QLabel("Shape: ???")
        layout.addWidget(self.shape)
        self.setLayout(layout)

    def set_shape(self, shape: Tuple[int, int]) -> None:
        """Set the shape to show in the labels.

        shape : Tuple[int, int]
            The shape to show in the labels.
        """
        self.shape.setText(f"Shape: ({shape[0]}, {shape[1]})")


class QtTestImageLayout(QVBoxLayout):
    """Controls to a create a new test image layer.

    Parameters
    ----------
    on_create : Callable[[], None]
        Called when the create test image button is pressed.
    """

    def __init__(self, on_create: Callback):
        super().__init__()
        self.addStretch(1)

        self.name = QComboBox()
        self.name.addItems(TEST_IMAGES.keys())
        self.name.activated[str].connect(self._on_name)
        self.addWidget(self.name)

        ShapeControls = namedtuple('ShapeControls', "set fixed")
        self.shape_controls = ShapeControls(QtSetShape(), QtFixedShape())

        # Add both, but only one will be visible at a time.
        self.addWidget(self.shape_controls.set)
        self.addWidget(self.shape_controls.fixed)

        # User can always set the tile size. Tiles always square for now.
        self.tile_size = QtLabeledSpinBox(
            "Tile Size", TILE_SIZE_DEFAULT, TILE_SIZE_RANGE
        )
        self.addLayout(self.tile_size)

        # Checkbox so we can choose between OctreeImage and regular Image.
        self.octree = QCheckBox("Octree Image")
        self.octree.setChecked(1)
        self.addWidget(self.octree)

        # The create button.
        button = QPushButton("Create Test Image")
        button.setToolTip("Create a new test image")
        button.clicked.connect(on_create)
        self.addWidget(button)

        # Set the initially selected image.
        self._on_name("Digits")

    def _on_name(self, value: str) -> None:
        """Called when a new image name is selected.

        Set which image controls are visible based on the spec of
        the newly selected image.

        Parameters
        ----------
        value : str
            The new image name.
        """
        spec = TEST_IMAGES[value]

        if spec['shape'] is None:
            # Image has a settable shape.
            self.shape_controls.set.show()
            self.shape_controls.fixed.hide()
        else:
            # Image has a fixed shape.
            self.shape_controls.set.hide()
            self.shape_controls.fixed.show()
            self.shape_controls.fixed.set_shape(spec['shape'])

    def get_image_shape(self) -> Tuple[int, int]:
        """Return the configured image shape.

        Return
        ------
        Tuple[int, int]
            The [height, width] shape requested by the user.
        """
        return self.shape_controls.set.get_shape()

    def get_tile_size(self) -> int:
        """Return the configured tile size.

        Return
        ------
        int
            The requested tile size.
        """
        return self.tile_size.spin.value()


class QtTestImage(QFrame):
    """Frame with controls to create a new test image.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer.
    """

    # Class attribute so system-wide we create unique names, even if
    # created from different QtRender widgets for different layers.
    image_index = 0

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.layout = QtTestImageLayout(self._create_test_image)
        self.setLayout(self.layout)

    def _create_test_image(self) -> None:
        """Create a new test image layer."""

        # Get the spec for the current selected type of image.
        image_name = self.layout.name.currentText()
        spec = TEST_IMAGES[image_name]
        factory = spec['factory']

        if spec['shape'] is None:
            # Image has a settable shape provided by the UI.
            shape = self.layout.get_image_shape()
            data = factory(shape)
        else:
            # Image comes in just one specific shape.
            data = factory()

        # Give each layer a unique name.
        unique_name = f"test-image-{QtTestImage.image_index:003}"
        QtTestImage.image_index += 1

        # Set config to create Octree or regular images.
        config.create_octree_images = self.layout.octree.isChecked()

        # Add the new image layer.
        layer = self.viewer.add_image(data, rgb=True, name=unique_name)
        layer.tile_size = self.layout.get_tile_size()
