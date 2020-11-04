from abc import ABC, abstractmethod

import numpy as np
from vispy.visuals.transforms import (
    ChainTransform,
    MatrixTransform,
    NullTransform,
    STTransform,
)

from .utils_gl import get_max_texture_sizes


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
        self._array_like = False
        self.node = node

        (
            self.MAX_TEXTURE_SIZE_2D,
            self.MAX_TEXTURE_SIZE_3D,
        ) = get_max_texture_sizes()

        self.layer.events.refresh.connect(lambda e: self.node.update())
        self.layer.events.set_data.connect(self._on_data_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.opacity.connect(self._on_opacity_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.scale.connect(self._on_matrix_change)
        self.layer.events.translate.connect(self._on_matrix_change)
        self.layer.events.rotate.connect(self._on_matrix_change)
        self.layer.events.shear.connect(self._on_matrix_change)
        self.layer.events.affine.connect(self._on_matrix_change)

    @property
    def _master_transform(self):
        """vispy.visuals.transforms.MatrixTransform:

        Central node's firstmost transform.
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if isinstance(self.node.transform, NullTransform):
            self.node.transform = ChainTransform(
                [STTransform(), MatrixTransform()]
            )

        return self.node.transform.transforms[1]

    @property
    def _grid_transform(self):
        """vispy.visuals.transforms.MatrixTransform:

        Transform used if viewer is in grid mode
        """
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, MatrixTransform):
            self.node.transform = ChainTransform(
                [STTransform(), MatrixTransform()]
            )

        return self.node.transform.transforms[0]

    def subplot(self, position, size):
        """Shift a layer to a specified position in a 2D grid.

        Parameters
        ----------
        position : 2-tuple of int
            New position of layer in grid (row, column).
        size : 2-tuple
            Size of grid cell (height, width).
        """
        # Determine translation
        translate = np.multiply(size, position)
        # Convert from NumPy ordering to Vispy ordering
        translate = translate[::-1]
        # Pad to make a 4-vector for Vispy
        padded_translate = np.pad(
            translate,
            ((0, 4 - len(translate))),
            constant_values=1,
            mode='constant',
        )

        # Check if translation value requires update
        if self.translate_grid is not None and np.all(
            self.translate_grid == padded_translate
        ):
            return
        self._grid_transform.translate = padded_translate

    @property
    def translate_grid(self):
        """sequence of float: Translation values for grid offset."""
        return self._grid_transform.translate

    @property
    def translate(self):
        """sequence of float: Translation values."""
        return self._master_transform.matrix[-1, :]

    @property
    def scale(self):
        """sequence of float: Scale factors."""
        matrix = self._master_transform.matrix[:-1, :-1]
        _, upper_tri = np.linalg.qr(matrix)
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

    def _on_matrix_change(self, event=None):
        transform = self.layer._transforms.simplified.set_slice(
            self.layer._dims.displayed
        )
        # convert NumPy axis ordering to VisPy axis ordering
        # by reversing the axes order and flipping the linear
        # matrix
        translate = transform.translate[::-1]
        matrix = transform.linear_matrix[::-1, ::-1].T

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        affine_matrix[-1, : len(translate)] = translate

        if self._array_like and self.layer._dims.ndisplay == 2:
            # Perform pixel offset to shift origin from top left corner
            # of pixel to center of pixel.
            # Note this offset is only required for array like data in
            # 2D.
            offset_matrix = (
                self.layer._transforms['data2world']
                .set_slice(self.layer._dims.displayed)
                .linear_matrix
            )
            offset = -offset_matrix @ np.ones(offset_matrix.shape[1]) / 2
            # Convert NumPy axis ordering to VisPy axis ordering
            # and embed in full affine matrix
            affine_offset = np.eye(4)
            affine_offset[-1, : len(offset)] = offset[::-1]
            affine_matrix = affine_matrix @ affine_offset
        self._master_transform.matrix = affine_matrix

    def _reset_base(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()

    def _on_camera_move(self, event=None):
        """Camera was moved."""
        pass
