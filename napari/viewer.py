from .components import Window, Viewer


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
        self.viewer_widget = Viewer()
        self.window = Window(self.viewer_widget)
        self.layers = self.viewer_widget.layers
        for image in images:
            self.add_image(image)
        for name, image in named_images.items():
            self.add_image(image, name=name)

    def add_layer(self, layer):
        self.viewer_widget.add_layer(layer)

    def remove_layer(self, layer_id):
        self.viewer_widget.layers.remove(layer_id)
