"""Basic example of using magicgui to create an Image Arithmetic GUI in napari."""
import enum
import numpy as np
import napari


# Enums are a convenient way to get a dropdown menu
class Operation(enum.Enum):
    """A set of valid arithmetic operations for image_arithmetic."""
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


# Define our function.
# Note that we can use forward references for the napari objects.
# You can read more about them here https://www.python.org/dev/peps/pep-0484/#forward-references
def image_arithmetic(layerA: 'napari.layers.Image', operation: Operation, layerB: 'napari.layers.Image') -> 'napari.layers.Image':
    """Adds, subtracts, multiplies, or divides two image layers of similar shape."""
    return operation.value(layerA.data, layerB.data)


# We also use the additional `call_button` option to add a button that
# will trigger function execution.
magic = {'call_button': "execute"}


with napari.gui_qt():
    # create a new viewer with a couple image layers
    viewer = napari.Viewer()
    viewer.add_image(np.random.rand(20, 20), name="Layer 1")
    viewer.add_image(np.random.rand(20, 20), name="Layer 2")

    # Add our magic function to napari
    viewer.window.add_function_widget(image_arithmetic, magic_kwargs=magic)
