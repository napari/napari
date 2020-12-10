"""Example showing how to accomplish a napari parameter sweep with magicgui.

It demonstrates:
1. overriding the default widget type with a custom class
2. the `auto_call` option, which calls the function whenever a parameter changes
"""
import napari
import skimage.data
import skimage.filters


# Define our gaussian blur function.
def gaussian_blur(layer: napari.layers.Image, sigma: float = 1.0, mode="nearest") -> napari.layers.Image:
    """Apply a gaussian blur to ``layer``."""
    if layer:
        return skimage.filters.gaussian(layer.data, sigma=sigma, mode=mode)


# Define our magic
magic = {
    'auto_call' : True,
    'sigma' : {"widget_type": "FloatSlider", "maximum": 6},
    'mode' : {"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
}


with napari.gui_qt():
    # create a viewer and add some images
    viewer = napari.Viewer()
    viewer.add_image(skimage.data.astronaut().mean(-1), name="astronaut")
    viewer.add_image(skimage.data.grass().astype("float"), name="grass")

    # Add our magic function to napari
    viewer.window.add_function_widget(gaussian_blur, magic_kwargs=magic)
