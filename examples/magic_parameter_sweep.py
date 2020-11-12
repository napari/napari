"""Example showing how to accomplish a napari parameter sweep with magicgui.

It demonstrates:
1. overriding the default widget type with a custom class
2. the `auto_call` option, which calls the function whenever a parameter changes
"""
import napari
import skimage.data
import skimage.filters
from magicgui._qt.widgets import QDoubleSlider


# Define our gaussian blur function.
def gaussian_blur(layer: napari.layers.Image, sigma: float = 1.0, mode="nearest") -> napari.layers.Image:
    """Apply a gaussian blur to ``layer``."""
    if layer:
        return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)


# Define our magic
# - `auto_call` tells magicgui to call the function whenever a parameter changes
# - we use `widget_type` to override the default "float" widget on sigma
# - we provide some Qt-specific parameters
# - we contstrain the possible choices for `mode`
magic = {
    'auto_call' : True,
    'sigma' : {"widget_type": QDoubleSlider, "maximum": 6, "fixedWidth": 400},
    'mode' : {"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
}


with napari.gui_qt():
    # create a viewer and add some images
    viewer = napari.Viewer()
    viewer.add_image(skimage.data.astronaut().mean(-1), name="astronaut")
    viewer.add_image(skimage.data.grass().astype("float"), name="grass")

    # Add our magic function to napari
    viewer.window.add_magic_function(gaussian_blur, magic)
