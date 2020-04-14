from abc import ABC, abstractmethod
from functools import lru_cache
import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl
from vispy.visuals.transforms import STTransform


class VispyBaseLayer(ABC):
    """Base object for individual layer views

    Meant to be subclassed.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.

    Attributes
    ----------
    layer : napari.layers.Layer
        Layer model.
    node : vispy.scene.VisualNode
        Central node with which to interact with the visual.
    scale : sequence of float
        Scale factors for the layer visual in the scenecanvas.
    translate : sequence of float
        Translation values for the layer visual in the scenecanvas.
    scale_factor : float
        Conversion factor from canvas coordinates to image coordinates, which
        depends on the current zoom level.
    MAX_TEXTURE_SIZE_2D : int
        Max texture size allowed by the vispy canvas during 2D rendering.
    MAX_TEXTURE_SIZE_3D : int
        Max texture size allowed by the vispy canvas during 2D rendering.

    Extended Summary
    ----------
    _master_transform : vispy.visuals.transforms.STTransform
        Transform positioning the layer visual inside the scenecanvas.
    """

    def __init__(self, layer, node):
        super().__init__()

        self.layer = layer
        self.node = node

        MAX_TEXTURE_SIZE_2D, MAX_TEXTURE_SIZE_3D = get_max_texture_sizes()
        self.MAX_TEXTURE_SIZE_2D = MAX_TEXTURE_SIZE_2D
        self.MAX_TEXTURE_SIZE_3D = MAX_TEXTURE_SIZE_3D

        self._position = (0,) * self.layer.dims.ndisplay

        self.layer.events.refresh.connect(lambda e: self.node.update())
        self.layer.events.set_data.connect(self._on_data_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.opacity.connect(self._on_opacity_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.scale.connect(self._on_scale_change)
        self.layer.events.translate.connect(self._on_translate_change)

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.STTransform:
        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, STTransform):
            self.node.transform = STTransform()

        return self.node.transform

    @property
    def order(self):
        """int: Order in which the visual is drawn in the scenegraph.

        Lower values are closer to the viewer.
        """
        return self.node.order

    @order.setter
    def order(self, order):
        self.node.order = order

    @property
    def scale(self):
        """sequence of float: Scale factors."""
        return self._master_transform.scale

    @scale.setter
    def scale(self, scale):
        # Avoid useless update if nothing changed in the displayed dims
        # Note that the master_transform scale is always a 4-vector so pad
        padded_scale = np.pad(scale, ((0, 4 - len(scale))), constant_values=1)
        if self.scale is not None and np.all(self.scale == padded_scale):
            return
        self._master_transform.scale = padded_scale

    @property
    def translate(self):
        """sequence of float: Translation values."""
        return self._master_transform.translate

    @translate.setter
    def translate(self, translate):
        # Avoid useless update if nothing changed in the displayed dims
        # Note that the master_transform translate is always a 4-vector so pad
        padded_translate = np.pad(
            translate, ((0, 4 - len(translate))), constant_values=1
        )
        if self.translate is not None and np.all(
            self.translate == padded_translate
        ):
            return
        self._master_transform.translate = padded_translate

    @property
    def scale_factor(self):
        """float: Conversion factor from canvas coordinates to image
        coordinates, which depends on the current zoom level.
        """
        transform = self.node.canvas.scene.node_transform(self.node)
        scale_factor = transform.map([1, 1])[0] - transform.map([0, 0])[0]
        return scale_factor

    @abstractmethod
    def _on_data_change(self, event=None):
        raise NotImplementedError()

    def _on_visible_change(self, event=None):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self, event=None):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self, event=None):
        self.node.set_gl_state(self.layer.blending)
        self.node.update()

    def _on_scale_change(self, event=None):
        scale = self.layer._transforms.simplified.set_slice(
            self.layer.dims.displayed
        ).scale
        # convert NumPy axis ordering to VisPy axis ordering
        self.scale = scale[::-1]
        if self.layer.is_pyramid:
            self.layer.corner_pixels, size = self.find_extrema()
        self.layer.position = self._transform_position(self._position)

    def _on_translate_change(self, event=None):
        translate = self.layer._transforms.simplified.set_slice(
            self.layer.dims.displayed
        ).translate
        # convert NumPy axis ordering to VisPy axis ordering
        self.translate = translate[::-1]
        self.layer.position = self._transform_position(self._position)

    def _transform_position(self, position):
        """Transform cursor position from canvas space (x, y) into image space.

        Parameters
        -------
        position : 2-tuple
            Cursor position in canvase (x, y).

        Returns
        -------
        coords : tuple
            Coordinates of cursor in image space for displayed dimensions only
        """
        nd = self.layer.dims.ndisplay
        if self.node.canvas is not None:
            transform = self.node.canvas.scene.node_transform(self.node)
            # Map and offset position so that pixel center is at 0
            mapped_position = transform.map(list(position))[:nd] - 0.5
            coords = tuple(mapped_position[::-1])
        else:
            coords = (0,) * nd
        return coords

    def _reset_base(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_scale_change()
        self._on_translate_change()

    def compute_data_level(self, size, size_threshold=None):
        """Computed desired level of the pyramid given requested field of view.

        The level of the pyramid should be the highest resolution such that
        the largest axis of the displayed slice is less than the size
        threshold.

        Parameters
        ----------
        size : tuple
            Requested size of field of view in data coordinates
        size_threshold : int, optional
            Maximum size of a displayed tile in pixels. If not provided then
            taken from the size of the canvas.

        Returns
        ----------
        level : int
            Level of the pyramid to be viewing.
        """

        # Determine size threshold
        if size_threshold is None:
            size_threshold = np.max(self.node.canvas.size)

        # Only consider displayed dimensions
        size = size[self.layer.dims.displayed]
        ds = self.layer.level_downsamples[:, self.layer.dims.displayed]

        # Consider largest dimension after scaling by downsamples
        ratio = (size / ds).max(axis=1)

        # Find the maximum allowed size
        level = np.argmin(ratio > size_threshold)

        return level

    def find_extrema(self):
        """Finds the extreme pixels of the canvas in data.

        Depends on the current pan and zoom position.

        Returns
        ----------
        corner_pixels : array
            Coordinates of top left and bottom right canvas pixel in the data.
        size : array
            Size of requested tile in data coordinates.
        """
        nd = self.layer.dims.ndisplay
        # Find image coordinate of top left canvas pixel
        if self.node.canvas is not None:
            offset = self.translate[:nd] / self.scale[:nd]
            tl_raw = np.floor(self._transform_position([0, 0]) + offset[::-1])
            br_raw = np.ceil(
                self._transform_position(self.node.canvas.size) + offset[::-1]
            )
        else:
            tl_raw = [0] * nd
            br_raw = [1] * nd

        top_left = np.zeros(self.layer.ndim)
        bottom_right = np.zeros(self.layer.ndim)
        for d, tl, br in zip(self.layer.dims.displayed, tl_raw, br_raw):
            top_left[d] = tl
            # take an extra pixel from the bottom right to prevent artifacts
            # at the bottom and right edges of the canvas
            bottom_right[d] = br + 1

        # Clip according to the max data of the level shape
        top_left = np.clip(
            top_left,
            0,
            np.subtract(self.layer.level_shapes[self.layer.data_level], 1),
        )
        bottom_right = np.clip(
            bottom_right,
            0,
            np.subtract(self.layer.level_shapes[self.layer.data_level], 1),
        )

        # Scale to full resolution of the data
        size = (bottom_right - top_left) * self.layer.level_downsamples[
            self.layer.data_level
        ]

        return np.array([top_left, bottom_right]).astype(int), size

    def on_draw(self, event):
        """Called whenever the canvas is drawn, which happens whenever new
        data is sent to the canvas or the camera is moved.
        """
        self.layer.scale_factor = self.scale_factor
        if self.layer.is_pyramid:
            corner_pixels, size = self.find_extrema()
            data_level = self.compute_data_level(size)

            if data_level != self.layer.data_level:
                # Set the data level, which will trigger further updates
                # including recaluation of the corner_pixels for the new
                # level
                self.layer.data_level = data_level
            else:
                self.layer.corner_pixels = corner_pixels


@lru_cache()
def get_max_texture_sizes():
    """Get maximum texture sizes for 2D and 3D rendering.

    Returns
    -------
    MAX_TEXTURE_SIZE_2D : int or None
        Max texture size allowed by the vispy canvas during 2D rendering.
    MAX_TEXTURE_SIZE_3D : int or None
        Max texture size allowed by the vispy canvas during 2D rendering.
    """
    # A canvas must be created to access gl values
    c = Canvas(show=False)
    try:
        MAX_TEXTURE_SIZE_2D = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)
    finally:
        c.close()
    if MAX_TEXTURE_SIZE_2D == ():
        MAX_TEXTURE_SIZE_2D = None
    # vispy doesn't expose GL_MAX_3D_TEXTURE_SIZE so hard coding
    # MAX_TEXTURE_SIZE_3D = gl.glGetParameter(gl.GL_MAX_3D_TEXTURE_SIZE)
    # if MAX_TEXTURE_SIZE_3D == ():
    #    MAX_TEXTURE_SIZE_3D = None
    MAX_TEXTURE_SIZE_3D = 2048

    return MAX_TEXTURE_SIZE_2D, MAX_TEXTURE_SIZE_3D
