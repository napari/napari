from .components import Window, Viewer


class ViewerApp(Viewer):
    """Napari ndarray viewer.

    Parameters
    ----------
    *images : ndarray
        Arrays to render as image layers.
    meta : dictionary, optional
        A dictionary of metadata attributes. If multiple images are provided,
        the metadata applies to all of them.
    multichannel : bool, optional
        Whether to consider the last dimension of the image(s) as channels
        rather than spatial attributes. If not provided, napari will attempt
        to make an educated guess. If provided, and multiple images are given,
        the same value applies to all images.
    clim_range : list | array | None
        Length two list or array with the default color limit range for the
        image. If not passed will be calculated as the min and max of the
        image. Passing a value prevents this calculation which can be useful
        when working with very large datasets that are dynamically loaded.
        If provided, and multiple images are given, the same value applies to
        all images.
    **named_images : dict of str -> ndarray, optional
        Arrays to render as image layers, keyed by layer name.
    """
    def __init__(self, *images, meta=None, multichannel=None, clim_range=None,
                 **named_images):
        super().__init__()
        self.window = Window(self)
        for image in images:
            self.add_image(image, meta=meta, multichannel=multichannel,
                           clim_range=clim_range)
        for name, image in named_images.items():
            self.add_image(image, meta=meta, multichannel=multichannel,
                           clim_range=clim_range, name=name)
