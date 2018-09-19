from vispy.visuals.transforms import STTransform
from vispy.visuals.filters import Alpha

from ..visuals.napari_image import NapariImage as Image

from ..util import (is_multichannel,
                    guess_multichannel,
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
    """
    def __init__(self, image, meta, view):
        self.image = image
        self.meta = meta
        self.view = view

        self.visual = Image(image, parent=view.scene,
                            method='auto')

        self._alpha = Alpha(1.0)
        self.visual.attach(self._alpha)
        self._interpolation_index = 0

        self.interpolation = 'nearest'

    def __str__(self):
        """Gets the image title."""
        info = ['image']

        try:
            info.append(self.meta['name'])
        except KeyError:
            pass

        info.append(self.image.shape)
        info.append(self.interpolation)

        return ' '.join(str(x) for x in info)

    def __repr__(self):
        """Equivalent to str(obj).
        """
        return 'ImageContainer: ' + str(self)

    def set_view_slice(self, indices):
        """Sets the view given the indices to slice.

        Parameters
        ----------
        indices : list
            Indices to slice with.
        """
        self.indices = indices
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

        self.visual.set_data(sliced_image)
        self.view.camera.set_range()

    def change_image(self, image, meta=None, multichannel=None):
        """Changes the underlying image and metadata.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata. If None, reuses previous metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        """
        self.image = image

        if meta is not None:
            self.meta = meta

        if multichannel is None:
            multichannel = guess_multichannel(image.shape)

        if multichannel:
            self.meta['itype'] = 'multi'

        self.visual._need_colortransform_update = True
        self.set_view_slice(self.indices)
        self.visual.update()

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return _index_to_name(self.interpolation_index)

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
        self.visual.interpolation = _index_to_name(intp_index)

    @property
    def transparency(self):
        """float: Image transparency.
        """
        return self._alpha.alpha

    @transparency.setter
    def transparency(self, transparency):
        if not 0.0 <= transparency <= 1.0:
            raise ValueError('transparency must be between 0-1, '
                             f'not {transparency}')

        self._alpha.alpha = transparency
        self.visual.update()

    @property
    def cmap(self):
        """string: Color map.
        """
        return self.visual.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.visual.cmap = cmap

    @property
    def transform(self):
        """vispy.visuals.transform.BaseTransform: Transformation.
        """
        return self.visual.transform

    @transform.setter
    def transform(self, transform):
        self.visual.transform = transform
