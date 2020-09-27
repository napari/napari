from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl
from vispy.visuals.transforms import MatrixTransform


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
    ----------------
    _master_transform : vispy.visuals.transforms.MatrixTransform
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
        self.layer.events.scale.connect(self._on_matrix_change)
        self.layer.events.translate.connect(self._on_matrix_change)
        self.layer.events.rotate.connect(self._on_matrix_change)
        self.layer.events.shear.connect(self._on_matrix_change)
        self.layer.events.loaded.connect(self._on_loaded_change)

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.MatrixTransform:
        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, MatrixTransform):
            self.node.transform = MatrixTransform()

        return self.node.transform

    @property
    def translate(self):
        """sequence of float: Translation values."""
        return self._master_transform.matrix[:, -1]

    @property
    def scale(self):
        """sequence of float: Scale factors."""
        matrix = self._master_transform.matrix[:-1, :-1]
        upper_tri = np.linalg.cholesky(np.dot(matrix.T, matrix)).T
        return np.diag(upper_tri).copy()

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
    def scale_factor(self):
        """float: Conversion factor from canvas pixels to data coordinates.
        """
        if self.node.canvas is not None:
            transform = self.node.canvas.scene.node_transform(self.node)
            return transform.map([1, 1])[0] - transform.map([0, 0])[0]
        else:
            return 1

    @abstractmethod
    def _on_data_change(self, event=None):
        raise NotImplementedError()

    def _on_visible_change(self, event=None):
        self.node.visible = self.layer.visible and self.layer.loaded

    def _on_opacity_change(self, event=None):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self, event=None):
        self.node.set_gl_state(self.layer.blending)
        self.node.update()

    def _on_matrix_change(self, event=None):
        transform = self.layer._transforms.simplified.set_slice(
            self.layer.dims.displayed
        )
        # convert NumPy axis ordering to VisPy axis ordering
        matrix = transform.linear_matrix[::-1, ::-1]
        translate = transform.translate[::-1]
        # Embed in 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        affine_matrix[-1, : len(translate)] = translate
        self._master_transform.matrix = affine_matrix
        self.layer.corner_pixels = self.coordinates_of_canvas_corners()
        self.layer.position = self._transform_position(self._position)

    def _on_loaded_change(self, event=None):
        self.node.visible = self.layer.visible and self.layer.loaded

    def _transform_position(self, position):
        """Transform cursor position from canvas space (x, y) into image space.

        Parameters
        ----------
        position : 2-tuple
            Cursor position in canvas (x, y).

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
            return tuple(mapped_position[::-1])
        else:
            return (0,) * nd

    def _reset_base(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()

    def coordinates_of_canvas_corners(self):
        """Find location of the corners of canvas in data coordinates.

        This method should only be used during 2D image viewing. The result
        depends on the current pan and zoom position.

        Returns
        -------
        corner_pixels : array
            Coordinates of top left and bottom right canvas pixel in the data.
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
            bottom_right[d] = br

        return np.array([top_left, bottom_right]).astype(int)

    def on_draw(self, event):
        """Called whenever the canvas is drawn.

        This is triggered from vispy whenever new data is sent to the canvas or
        the camera is moved and is connected in the `QtViewer`.
        """
        self.layer.scale_factor = self.scale_factor
        old_corner_pixels = self.layer.corner_pixels
        self.layer.corner_pixels = self.coordinates_of_canvas_corners()

        # For 2D multiscale data determine if new data has been requested
        if (
            self.layer.multiscale
            and self.layer.dims.ndisplay == 2
            and self.node.canvas is not None
        ):
            self.layer._update_multiscale(
                corner_pixels=old_corner_pixels,
                shape_threshold=self.node.canvas.size,
            )


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
