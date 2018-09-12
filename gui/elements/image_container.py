from vispy import scene
from ..visuals.napari_image import NapariImage

from ..util import is_rgb


# get available interpolation methods
interpolation_method_names = scene.visuals.Image(None).interpolation_functions
interpolation_method_names = list(interpolation_method_names)
interpolation_method_names.sort()
interpolation_method_names.remove('sinc')  # does not work well on my machine

# print(interpolation_method_names)
index_to_name = interpolation_method_names.__getitem__
name_to_index = interpolation_method_names.index


class ImageContainer:
    """Image container.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict
        Image metadata.
    view : vispy.scene.widgets.ViewBox
        View on which to draw.
    update_func : callable
        Function to call when the image needs to be redrawn.
        Takes no arguments.
    """
    def __init__(self, image, meta, view, update_func):
        self.image = image
        self.meta = meta
        self.view = view
        self.update = update_func

        self.image_visual = NapariImage(image, parent=view.scene,
                                        method='auto')

        self._brightness = 1
        self._interpolation_index = 0

        self.interpolation = 'nearest'
        for k, v in self.meta.__dict__.items():
            self.update_from_metadata(k, v)

        self.meta.update_hooks.append(self.update_from_metadata)

    def __str__(self):
        """Gets the image title."""
        info = ['image']

        try:
            info.append(self.meta.name)
        except AttributeError:
            pass

        info.append(self.image.shape)
        info.append(self.interpolation)

        return ' '.join(str(x) for x in info)

    def update_from_metadata(self, name, value):
        try:
            setattr(self, name, value)
        except AttributeError:
            pass

    def set_view(self, indices):
        """Sets the view given the indices to slice.

        Parameters
        ----------
        indices : list
            Indices to slice with.
        """
        ndim = self.image.ndim - is_rgb(self.meta)
        indices = indices[:ndim]

        for dim in range(len(indices)):
            dim_len = self.image.shape[dim]

            try:
                if indices[dim] > dim_len:
                    indices[dim] = dim_len
            except TypeError:
                pass

        sliced_image = self.image[tuple(indices)]

        self.image_visual.set_data(sliced_image)
        self.view.camera.set_range()

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = name_to_index(interpolation)

    @property
    def interpolation_index(self):
        """int: Index of the current interpolation method equipped.
        """
        return self._interpolation_index

    @interpolation_index.setter
    def interpolation_index(self, interpolation_index):
        intp_index = interpolation_index % len(interpolation_method_names)
        self._interpolation_index = intp_index
        self.image_visual.interpolation = index_to_name(intp_index)
        # print(self.image_visual.interpolation)
        self.update()

    @property
    def brightness(self):
        """float: Image brightness.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, brightness):
        # TODO: actually implement this
        print("brightess = %f" % brightness)
        if not 0.0 < brightness < 1.0:
            raise ValueError('brightness must be between 0-1, not '
                             + brightness)

        self.brightness = brightness
        self.update()

    @property
    def cmap(self):
        """string: Color map.
        """
        return self.image_visual.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.image_visual.cmap = cmap
        self.update()
