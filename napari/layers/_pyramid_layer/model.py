import numpy as np
from copy import copy

from ...util.event import Event

from .._register import add_to_viewer
from .._image_layer import Image
from .._image_layer.view import QtImageLayer
from .._image_layer.view import QtImageControls


@add_to_viewer
class Pyramid(Image):
    """Image pyramid layer.

    Parameters
    ----------
    pyramid : list
        List of np.ndarry image data, with base of pyramid at `0`.
    meta : dict, optional
        Image metadata.
    multichannel : bool, optional
        Whether the image is multichannel. Guesses if None.
    name : str, keyword-only
        Name of the layer.
    clim_range : list | array | None
        Length two list or array with the default color limit range for the
        image. If not passed will be calculated as the min and max of the
        image. Passing a value prevents this calculation which can be useful
        when working with very large datasets that are dynamically loaded.
    **kwargs : dict
        Parameters that will be translated to metadata.
    """

    def __init__(self, pyramid, meta=None, multichannel=None, *, name=None,
                 clim_range=None, **kwargs):

        self._pyramid = pyramid
        self._image_shapes = np.array([im.shape[:2] for im in pyramid])
        avg_shape = self._image_shapes.max(axis=1)
        self._image_downsamples = avg_shape[0]/avg_shape
        self._pyramid_level = len(pyramid)-1

        self._max_tile_shape = np.array([1200, 1200])
        self._top_left = np.array([0, 0])

        super().__init__(pyramid[self._pyramid_level], meta=meta,
                         multichannel=multichannel, name=name,
                         clim_range=clim_range, **kwargs)
        self.scale = [self._image_downsamples[self._pyramid_level]] * 2

    @property
    def pyramid(self):
        """list: List of np.ndarry image data, with base of pyramid at `0`.
        """
        return self._pyramid

    @pyramid.setter
    def pyramid(self, pyramid):
        self._pyramid = pyramid

        self.refresh()

    @property
    def pyramid_level(self):
        """int: Level of pyramid to show, with base of pyramid at `0`.
        """
        return self._pyramid_level

    @pyramid_level.setter
    def pyramid_level(self, level):
        if self._pyramid_level == level:
            return
        self._pyramid_level = level
        self.scale = [self._image_downsamples[self.pyramid_level]] * 2
        self._top_left = self.find_top_left()
        self._update_image_from_pyramid()
        self.refresh()

    @property
    def top_left(self):
        """int: Location of top left pixel.
        """
        return self._top_left

    @top_left.setter
    def top_left(self, top_left):
        if np.all(self._top_left == top_left):
            return
        self._top_left = top_left
        self._update_image_from_pyramid()
        self.refresh()

    def _update_image_from_pyramid(self):
        """Updates the image data from the pyramid based on requested tile
        and pyramid level
        """
        if np.any(self._image_shapes[self.pyramid_level] >
                  self._max_tile_shape):
            slices = [slice(self._top_left[i], self._top_left[i] +
                            self._max_tile_shape[i], 1) for i in range(2)]
            self._image = self.pyramid[self.pyramid_level][tuple(slices)]
            self.translate = self._top_left[::-1]*self.scale[:2]
        else:
            self._image = self.pyramid[self.pyramid_level]
            self.translate = [0, 0]

    def _get_shape(self):
        """Shape of base of pyramid

        Returns
        ----------
        shape : np.ndarry
            Shape of base of pyramid
        """
        if self.multichannel:
            return self.pyramid[0].shape[:-1]
        return self.pyramid[0].shape

    def get_value(self, position, indices):
        """Returns coordinates, values, and a string for a given mouse position
        and set of indices.

        Parameters
        ----------
        position : sequence of two int
            Position of mouse cursor in canvas.
        indices : sequence of int or slice
            Indices that make up the slice.

        Returns
        ----------
        coord : sequence of int
            Position of mouse cursor in data.
        value : int or float or sequence of int or float
            Value of the data at the coord.
        msg : string
            String containing a message that can be used as
            a status update.
        """
        self._pos = position
        transform = self._node.canvas.scene.node_transform(self._node)
        pos_in_tile = transform.map(position)[:2]
        pos_in_slice = pos_in_tile +  self.translate[:2]/self.scale[:2]

        # Make sure pos in slice doesn't go off edge
        shape = self._image_shapes[self._pyramid_level]
        pos_in_slice = ([np.clip(pos_in_slice[1], 0, shape[0]-1),
                         np.clip(pos_in_slice[0], 0, shape[1]-1)])

        # Get value of image from tile
        adj_coord = [int(pos_in_tile[1]), int(pos_in_tile[0])]
        adj_coord = np.clip(adj_coord, 0,
                            np.array(self._image_view.shape[:2])-1)
        value = self._image_view[tuple(adj_coord)]

        coord = copy(indices)
        coord[0] = int(pos_in_slice[0]*self.scale[0])
        coord[1] = int(pos_in_slice[1]*self.scale[1])

        msg = f'{coord}, {self.pyramid_level}, {self.name}' + ', value '
        if isinstance(value, np.ndarray):
            if isinstance(value[0], np.integer):
                msg = msg + str(value)
            else:
                v_str = '[' + str.join(', ', [f'{v:0.3}' for v in value]) + ']'
                msg = msg + v_str
        else:
            if isinstance(value, np.integer):
                msg = msg + str(value)
            else:
                msg = msg + f'{value:0.3}'
        return coord, value, msg

    def compute_pyramid_level(self, camera):
        """Computed what level of the pyramid should be viewed given the
        current camera position.

        Parameters
        ----------
        camera : Camera
            Vispy PanZoomCamera object.

        Returns
        ----------
        level : int
            Level of the pyramid to be viewing.
        """
        size = camera.rect.size
        level = np.round(np.clip(np.log2(np.array(size).min())-8, 0,
                                 len(self.pyramid)-1)).astype('int')
        return level

    def find_top_left(self):
        """Finds the top left pixel of the canvas. Depends on the current
        pan and zoom position

        Returns
        ----------
        top_left : np.ndarry
            Length two array with coordinates of top left pixel.
        """

        # Find image coordinate of top left canvas pixel
        transform = self._node.canvas.scene.node_transform(self._node)
        pos = transform.map([0, 0])[:2] + self.translate[:2]/self.scale[:2]

        shape = self._image_shapes[0]

        # Clip according to the max image shape
        pos = [np.clip(pos[1], 0, shape[0]-1),
               np.clip(pos[0], 0, shape[1]-1)]

        # Convert to offset for image array
        top_left = np.array(pos)
        scale = self._max_tile_shape/4
        top_left = scale*np.floor(top_left/scale)

        return top_left.astype(int)

    def on_draw(self, event):
        """Called whenever the canvas is drawn.
        """
        camera = self.viewer._qt.view.camera
        shape = self._image_shapes[self.pyramid_level]
        pyramid_level = self.compute_pyramid_level(camera)
        if pyramid_level != self.pyramid_level:
            self.pyramid_level = pyramid_level
        else:
            self.top_left = self.find_top_left()
