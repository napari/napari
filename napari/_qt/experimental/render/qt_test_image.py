"""QtTestImage class.

Creating test images is meant as an internal developer feature for now.
"""
from collections import namedtuple
from typing import Callable, Tuple

from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ....components.experimental.chunk import async_config
from ....layers.image.experimental import TestImageSettings
from ....utils import config
from .image_creator import create_test_image_multi
from .qt_render_widgets import QtLabeledComboBox, QtLabeledSpinBox

Callback = Callable[[], None]
IntCallback = Callable[[int], None]

TILE_SIZE_RANGE = range(1, 4096, 100)

IMAGE_SHAPE_DEFAULT = (1024, 1024)  # (height, width)
IMAGE_SHAPE_RANGE = range(1, 65536, 100)


# We can create 3 types of images layers.
IMAGE_TYPES = {
    "Normal": config.CREATE_IMAGE_NORMAL,
    "Compound": config.CREATE_IMAGE_COMPOUND,
    "Tiled": config.CREATE_IMAGE_TILED,
}
IMAGE_TYPE_DEFAULT = "Tiled"

# Allow the user to pick test images by name. If the "shape" is none then
# the user can chose the shape with two spin controls.
TEST_IMAGES = {
    "Digits": {
        "shape": None,
        "factory": lambda image_settings: create_test_image_multi(
            "0", image_settings
        ),
    },
}
TEST_IMAGE_DEFAULT = "Digits"

# Add skimage.data images if installed. Napari does not depend on
# skimage but many developers will have it.
try:
    import skimage.data

    TEST_IMAGES.update(
        {
            "Astronaut": {
                "shape": (512, 512),
                "factory": lambda: skimage.data.astronaut(),
            },
            "Chelsea": {
                "shape": (300, 451),
                "factory": lambda: skimage.data.chelsea(),
            },
            "Coffee": {
                "shape": (400, 600),
                "factory": lambda: skimage.data.coffee(),
            },
        }
    )
except ImportError:
    pass  # The skimage.data images won't be available.


class QtVariableShape(QGroupBox):
    """Two spin boxes used to set the shape of an image."""

    def __init__(self):
        super().__init__("Dimensions")

        layout = QVBoxLayout()
        self.height = QtLabeledSpinBox(
            "Height", IMAGE_SHAPE_DEFAULT[0], IMAGE_SHAPE_RANGE
        )
        self.width = QtLabeledSpinBox(
            "Width", IMAGE_SHAPE_DEFAULT[1], IMAGE_SHAPE_RANGE
        )
        layout.addWidget(self.height)
        layout.addWidget(self.width)
        self.setLayout(layout)

    def get_shape(self) -> Tuple[int, int]:
        """Return the currently configured shape.

        Return
        ------
        Tuple[int, int]
            The requested [height, width] shape.
        """
        return self.height.spin.value(), self.width.spin.value()


class QtFixedShape(QGroupBox):
    """A label to display the fixed shape of an image."""

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

        # Shows the test images we can create.
        self.name = QComboBox()
        self.name.addItems(TEST_IMAGES.keys())
        self.name.activated[str].connect(self._on_name)
        self.addWidget(self.name)

        # Create shape controls for "variable sized" or "fixed size" images.
        # Add both sets of controls, but only one set is visible at a time.
        ShapeControls = namedtuple('ShapeControls', "variable fixed")
        self.shape_controls = ShapeControls(QtVariableShape(), QtFixedShape())
        self.addWidget(self.shape_controls.variable)
        self.addWidget(self.shape_controls.fixed)

        # The tile size is available with either type of octree layer.
        self.tile_size = QtLabeledSpinBox(
            "Tile Size", async_config.octree.tile_size, TILE_SIZE_RANGE
        )
        self.addWidget(self.tile_size)

        # Which type of image layer/visual to create.
        self.image_type = QtLabeledComboBox("Type", IMAGE_TYPES)
        self.image_type.set_value(config.create_image_type)
        self.image_type.combo.activated[str].connect(self._on_type)
        self.addWidget(self.image_type)

        # The create button.
        button = QPushButton("Create Test Image")
        button.setToolTip("Create a new test image")
        button.clicked.connect(on_create)
        self.addWidget(button)

        # Set the initially selected image.
        self._on_name(TEST_IMAGE_DEFAULT)

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
            self.shape_controls.variable.show()
            self.shape_controls.fixed.hide()
        else:
            # Image has a fixed shape.
            self.shape_controls.variable.hide()
            self.shape_controls.fixed.show()
            self.shape_controls.fixed.set_shape(spec['shape'])

    def _on_type(self, _value: str) -> None:
        """User changed which type image they want to create.

        Parameters
        ----------
        value : str
            The new image type.
        """
        # Only show tile_size for octree-based images.
        self.tile_size.setVisible(
            self.image_type.get_value() != config.CREATE_IMAGE_NORMAL
        )

    @property
    def settings(self) -> TestImageSettings:
        """The desired image settings.

        Return
        ------
        ImageSetttings
            The desired image settings.
        """
        return TestImageSettings(
            self.shape_controls.variable.get_shape(),
            self.tile_size.spin.value(),
        )


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
        image_settings = self.layout.settings

        if spec['shape'] is None:
            # This image has a settable shape provided by the UI, so we
            # pass the settings into the factory.
            data = factory(image_settings)
        else:
            # This image comes in just one specific shape, so the factory
            # takes no arguments.
            data = factory()

        # Give each new layer a unique name.
        unique_name = f"test-image-{QtTestImage.image_index:003}"
        QtTestImage.image_index += 1

        # Set config for which type of image to create.
        image_type = self.layout.image_type.get_value()
        config.create_image_type = image_type

        # Add the new image layer.
        layer = self.viewer.add_image(data, rgb=True, name=unique_name)

        # TODO_OCTREE: We've not (yet?) added OctreeImage-specific
        # arguments to the OctreeImage constructor, because Layer class
        # arguments are special. They have to match the viewer add_image()
        # method?
        #
        # So for now we just set these values after construction, which is
        # kind of odd. And the class has to handle the value changing on
        # the fly. It would be better if they could be arguments or
        # passed in on construction somehow.
        layer.tile_size = image_settings.tile_size
