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
        self._image = image
        self._meta = meta
        self.view = view

        self.visual = Image(None, parent=view.scene,
                            method='auto')

        self._alpha = Alpha(1.0)
        self.visual.attach(self._alpha)
        self._interpolation_index = 0

        self.interpolation = 'nearest'

    def __str__(self):
        """Gets the image title.
        """
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
            max_dim_index = self.image.shape[dim] - 1

            try:
                if indices[dim] > max_dim_index:
                    indices[dim] = max_dim_index
            except TypeError:
                pass

        sliced_image = self.image[tuple(indices)]

        self.visual.set_data(sliced_image)
        self.update()

    def set_image(self, image, meta=None, multichannel=None):
        """Sets the underlying image and metadata.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata. If None, reuses previous metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        """
        self._image = image

        if meta is not None:
            self._meta = meta

        if multichannel is None:
            multichannel = guess_multichannel(image.shape)

        if multichannel:
            self.meta['itype'] = 'multi'

        self.refresh()

    def update(self):
        """Updates the underlying visual.
        """
        return self.visual.update()

    def refresh(self):
        """Fully refreshes the visual.
        """
        self.visual._need_colortransform_update = True
        self.set_view_slice(self.indices)
        self.update()

    @property
    def image(self):
        """np.ndarray: Image data.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.refresh()

    @property
    def meta(self):
        """dict: Image metadata.
        """
        # TODO: somehow listen for when metadata updates
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta
        self.refresh()

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
            raise ValueError('transparency must be between 0.0 and 1.0; '
                             f'got {transparency}')

        self._alpha.alpha = transparency
        self.update()

    ###
    ###  wrap visual properties
    ###

    @property
    def clim(self):
        """string or tuple of float: Limits to use for the colormap.
        Can be 'auto' to auto-set bounds to the min and max of the data.
        """
        return self.visual.clim

    @clim.setter
    def clim(self):
        self.visual.clim = clim

    @property
    def cmap(self):
        """string or ColorMap: Colormap to use for luminance images.
        """
        return self.visual.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.visual.cmap = cmap

    @property
    def method(self):
        """string: Selects method of rendering image in case of non-linear
        transforms. Each method produces similar results, but may trade
        efficiency and accuracy. If the transform is linear, this parameter
        is ignored and a single quad is drawn around the area of the image.

            * 'auto': Automatically select 'impostor' if the image is drawn
              with a nonlinear transform; otherwise select 'subdivide'.
            * 'subdivide': ImageVisual is represented as a grid of triangles
              with texture coordinates linearly mapped.
            * 'impostor': ImageVisual is represented as a quad covering the
              entire view, with texture coordinates determined by the
              transform. This produces the best transformation results, but may
              be slow.
        """
        return self.visual.method

    @method.setter
    def method(self):
        self.visual.method = method

    @property
    def size(self):
        """Size of the displayed image.
        """
        return self.visual.size

    @property
    def interpolation(self):
        """string: Equipped interpolation method's name.
        """
        return _index_to_name(self.interpolation_index)

    @interpolation.setter
    def interpolation(self, interpolation):
        self.interpolation_index = _name_to_index(interpolation)

    @property
    def interpolation_functions(self):
        return interpolation_names

    @property
    def transform(self):
        """vispy.visuals.transform.BaseTransform: Transformation.
        """
        return self.visual.transform

    @transform.setter
    def transform(self, transform):
        self.visual.transform = transform
