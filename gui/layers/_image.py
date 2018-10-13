import weakref

import numpy as np

from vispy.visuals.transforms import STTransform
from vispy.visuals.filters import Alpha

from .._vispy.scene.visuals import Image

from ..util import is_multichannel as _multichannel
from ..util.misc import guess_metadata
from ..util.interpolation import (interpolation_names,
                                  interpolation_index_to_name as _index_to_name,
                                  interpolation_name_to_index as _name_to_index)  # noqa


class ImageLayer:
    """Image layer.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    meta : dict
        Image metadata.
    viewer : Viewer
        Parent viewer widget.
    """
    def __init__(self, image, meta, viewer):
        self._image = image
        self._meta = meta
        self.viewer = viewer

        self.visual = Image(None, parent=viewer._qt.view.scene,
                            method='auto')

        self._interpolation_index = 0

        self.interpolation = 'nearest'

        self.indices = ...

        # TODO: implement and use STRTransform
        self.transform = STTransform()

        # update flags
        self._need_display_update = True
        self._need_visual_update = False

        self.update()

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
        return f'ImageLayer: {self} at {hex(id(self))}'

    def set_view_slice(self, indices):
        """Sets the view given the indices to slice with.

        Parameters
        ----------
        indices : sequence or ellipsis
            Indices to slice with.
        """
        self.indices = indices
        ndim = self.effective_ndim

        if indices is Ellipsis:
            indices = [0] * ndim
        else:
            indices = list(indices)
            indices = indices[:ndim]

            for dim in range(len(indices)):
                max_dim_index = self.image.shape[dim] - 1

                try:
                    if indices[dim] > max_dim_index:
                        indices[dim] = max_dim_index
                except TypeError:
                    pass

        indices[0] = slice(None)  # y-axis
        indices[1] = slice(None)  # x-axis
        sliced_image = self.image[tuple(indices)]

        self.visual.set_data(sliced_image)

        self._need_visual_update = True
        self.update()

    def set_image_and_metadata(self, image, meta):
        """Sets the underlying image and metadata.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict
            Image metadata.
        """
        self._image = image
        self._meta = meta

        self._need_display_update = True
        self.update()

    def imshow(self, image, meta=None, multichannel=None, **kwargs):
        """Replaces the current image with another.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        **kwargs : dict
            Parameters that will be translated to metadata.

        Returns
        -------
        self : ImageLayer
            Layer for the image.
        """
        meta = guess_metadata(image, meta, multichannel, kwargs)
        self.set_image_and_metadata(image, meta)
        return self

    def update(self):
        """Updates the underlying visual.
        """
        if self._need_display_update:
            self._need_display_update = False

            self.viewer._child_image_changed = True
            self.viewer.update()

            self.visual._need_colortransform_update = True
            self.set_view_slice(self.indices)

        if self._need_visual_update:
            self._need_visual_update = False
            self.visual.update()

    @property
    def viewer(self):
        viewer = self._viewer()
        if viewer is None:
            raise ValueError('Lost reference to viewer '
                             '(was garbage collected).')
        return viewer

    @viewer.setter
    def viewer(self, viewer):
        self._viewer = weakref.ref(viewer)

    @property
    def layout(self):
        """str: Layout type.
        """
        return self.viewer.layout

    @layout.setter
    def layout(self, layout):
        self.viewer.layout = layout

    @property
    def _layout(self):
        return self.viewer._layout

    @property
    def in_layout(self):
        """bool: If the layer is in a layout.
        """
        return self in self._layout

    @in_layout.setter
    def in_layout(self, in_layout):
        if in_layout == self.in_layout:
            return

        if self.in_layout:
            self._layout.remove_layer(self)
        else:
            self._layout.add_layer(self)

        self.viewer.reset_view()

    @property
    def image(self):
        """np.ndarray: Image data.
        """
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.viewer.update_sliders()

        self._need_display_update = True
        self.update()

    @property
    def meta(self):
        """dict: Image metadata.
        """
        # TODO: somehow listen for when metadata updates
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta

        self._need_display_update = True
        self.update()

    @property
    def data_ndim(self):
        """int: Number of dimensions of the array that
        represents the image data.
        """
        return self.image.ndim

    @property
    def effective_ndim(self):
        """int: Number of dimensions of the contained image.
        """
        return self.data_ndim - self.multichannel

    @property
    def display_ndim(self):
        """int: Number of dimensions being displayed in the visual.
        """
        return 2 + self.multichannel

    @property
    def extra_ndim(self):
        """int: Number of dimensions not displayed in the visual.
        """
        return self.data_ndim - self.display_ndim

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
    def multichannel(self):
        """bool: Whether the image is multichannel.
        """
        return _multichannel(self.meta)

    @multichannel.setter
    def multichannel(self, val):
        if val == self.multichannel:
            return

        self.meta['itype'] = 'multi'

        self._need_display_update = True
        self.update()

    @property
    def opacity(self):
        """float: Image opacity.
        """
        return self.visual.opacity

    @opacity.setter
    def opacity(self, opacity):
        if not 0.0 <= opacity <= 1.0:
            raise ValueError('opacity must be between 0.0 and 1.0; '
                             f'got {opacity}')

        self.visual.opacity = opacity

    @property
    def visible(self):
        """bool: Whether the visual is currently being displayed.
        """
        return self.visual.visible

    @visible.setter
    def visible(self, visibility):
        self.visual.visible = visibility

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
        if transform is None:
            transform = STTransform()
        self.visual.transform = transform

    ###
    ###  wrap STTransform attributes
    ###

    def _error_if_not_STTransform(self):
        tf = self.transform
        if not isinstance(tf, STTransform):
            raise TypeError('underlying transform expected to '
                            f'be STTransform; got {type(tf)}') from None

    @property
    def display_shape(self):
        return tuple(np.rint(self.scale[:2] * self.image.shape[:2],)
                     .astype(np.int))

    @property
    def scale(self):
        """sequence of float: Scale vector.
        """
        self._error_if_not_STTransform()
        return self.transform.scale

    @scale.setter
    def scale(self, vec):
        self._error_if_not_STTransform()
        self.transform.scale = vec

    @property
    def translate(self):
        """sequence of float: Translation vector.
        """
        self._error_if_not_STTransform()
        return self.transform.translate

    @translate.setter
    def translate(self, vec):
        self._error_if_not_STTransform()
        self.transform.translate = vec

    @property
    def z_index(self):
        """Image's ordering within the scene. A higher index means
        a higher rendering priority.

        Equivalent to .translate[2].
        """
        # add a check here if 3D-rendering is ever used.
        return -self.translate[2]

    @z_index.setter
    def z_index(self, index):
        # add a check here if 3D-rendering is ever used.
        self._error_if_not_STTransform()

        tf = self.transform
        tl = tf._translate
        tl[2] = -index

        tf._update_shaders()
        tf.update()
