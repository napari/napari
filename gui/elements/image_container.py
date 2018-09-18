from ..visuals.napari_image import NapariImage

from ..util import (is_multichannel,
                    interpolation_names,
                    interpolation_index_to_name as _index_to_name,
                    interpolation_name_to_index as _name_to_index)


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
        self._image = image
        self._meta = meta
        self.view = view
        self.update = update_func

        self.image_visual = NapariImage(image, parent=view.scene,
                                        method='auto')

        self._brightness = 1
        self._interpolation_index = 0

        self.interpolation = 'nearest'

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

    def __repr__(self):
        """Equivalent to str(obj).
        """
        return str(self)

    def set_view(self, indices):
        """Sets the view given the indices to slice.

        Parameters
        ----------
        indices : list
            Indices to slice with.
        """
        ndim = self.image.ndim - is_multichannel(self.meta)
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
    def image(self):
        """np.ndarray: Image data.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.update()

    @property
    def meta(self):
        """dict: Image metadata.
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta
        self.update()

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = _name_to_index(interpolation)

    @property
    def interpolation_index(self):
        """int: Index of the current interpolation method equipped.
        """
        return self._interpolation_index

    @interpolation_index.setter
    def interpolation_index(self, interpolation_index):
        intp_index = interpolation_index % len(interpolation_names)
        self._interpolation_index = intp_index
        self.image_visual.interpolation = _index_to_name(intp_index)
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
