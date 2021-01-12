"""Example showing how to accomplish a napari parameter sweep with magicgui.

It demonstrates:
1. overriding the default widget type with a custom class
2. the `auto_call` option, which calls the function whenever a parameter changes
"""
import skimage.data
import skimage.filters
import napari


# Define our gaussian_blur function.
# Note that we can use forward references for the napari type annotations.
# You can read more about them here:
# https://www.python.org/dev/peps/pep-0484/#forward-references
# In this example, because we have already imported napari anyway, it doesn't
# really matter. But this syntax would let you specify that a parameter is a
# napari object type without actually importing or depending on napari.
def gaussian_blur(
    layer: 'napari.layers.Image', sigma: float = 1.0, mode="nearest"
) -> 'napari.types.ImageData':
    """Apply a gaussian blur to ``layer``."""
    if layer:
        return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)


# Define our magic. The function will be automatically called when the
# input values are changed
magic = {
    'auto_call': True,
    'sigma': {"widget_type": "FloatSlider", "max": 6},
    'mode': {"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
}


with napari.gui_qt():
    # create a viewer and add some images
    viewer = napari.Viewer()
    viewer.add_image(skimage.data.astronaut().mean(-1), name="astronaut")
    viewer.add_image(skimage.data.grass().astype("float"), name="grass")

    # Add our magic function to napari
    viewer.window.add_function_widget(gaussian_blur, magic_kwargs=magic)
