from . import Window, Viewer


class ViewerApp:
    """Napari ndarray viewer.

    Parameters
    ----------
    *images : ndarray
        Arrays to render as image layers.
    **named_images : dict of str -> ndarray
        Arrays to render as image layers, keyed by layer name.
    """
    def __init__(self, *images, **named_images):
        v = Viewer()
        for image in images:
            v.add_image(image)
        for name, image in named_images.items():
            v.add_image(image, name=name)
        self.viewer_widget = v
        self.window = Window(v)
        self.layers = self.viewer_widget.layers

    def add_layer(self, layer):
        self.viewer_widget.add_layer(layer)
