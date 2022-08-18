import logging
from abc import ABC, abstractmethod

import numpy as np
from vispy.visuals.transforms import MatrixTransform

from napari.layers.base.base import _LayerSliceResponse
from napari.utils.transforms.transforms import Affine

from ...utils.events import disconnect_events
from ..utils.gl import BLENDING_MODES, get_max_texture_sizes

LOGGER = logging.getLogger("napari._vispy.layers.base")


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


    Notes
    -----
    _master_transform : vispy.visuals.transforms.MatrixTransform
        Transform positioning the layer visual inside the scenecanvas.
    """

    def __init__(self, layer, node):
        super().__init__()
        self.events = None  # Some derived classes have events.

        self.layer = layer
        self._array_like = False
        self.node = node

        (
            self.MAX_TEXTURE_SIZE_2D,
            self.MAX_TEXTURE_SIZE_3D,
        ) = get_max_texture_sizes()

        self.layer.events.refresh.connect(self._on_refresh_change)
        if not self.layer._is_async():
            self.layer.events.set_data.connect(self._on_data_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.opacity.connect(self._on_opacity_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.scale.connect(self._on_matrix_change)
        self.layer.events.translate.connect(self._on_matrix_change)
        self.layer.events.rotate.connect(self._on_matrix_change)
        self.layer.events.shear.connect(self._on_matrix_change)
        self.layer.events.affine.connect(self._on_matrix_change)
        self.layer.experimental_clipping_planes.events.connect(
            self._on_experimental_clipping_planes_change
        )

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
    def _on_data_change(self):
        raise NotImplementedError()

    def _on_refresh_change(self):
        self.node.update()

    # @abstractmethod # temporarily allow layers that don't implement this yet.
    def _set_slice(self, request: _LayerSliceResponse) -> None:
        raise NotImplementedError()

    def _on_visible_change(self):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self):
        blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**blending_kwargs)
        self.node.update()

    def _on_matrix_change(self):
        LOGGER.debug('VispyBaseLayer._on_matrix_change')
        transform = self.layer._transforms.simplified.set_slice(
            self.layer._dims_displayed
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

        if self._array_like and self.layer._ndisplay == 2:
            # Perform pixel offset to shift origin from top left corner
            # of pixel to center of pixel.
            # Note this offset is only required for array like data in
            # 2D.
            offset_matrix = self.layer._data_to_world.set_slice(
                self.layer._dims_displayed
            ).linear_matrix
            offset = -offset_matrix @ np.ones(offset_matrix.shape[1]) / 2
            # Convert NumPy axis ordering to VisPy axis ordering
            # and embed in full affine matrix
            affine_offset = np.eye(4)
            affine_offset[-1, : len(offset)] = offset[::-1]
            affine_matrix = affine_matrix @ affine_offset
        self._master_transform.matrix = affine_matrix

    def _on_experimental_clipping_planes_change(self):
        if hasattr(self.node, 'clipping_planes'):
            self.node.clipping_planes = (
                # invert axes because vispy uses xyz but napari zyx
                self.layer.experimental_clipping_planes.as_array()[..., ::-1]
            )

    def reset(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()
        self._on_experimental_clipping_planes_change()

    def _on_poll(self, event=None):
        """Called when camera moves, before we are drawn.

        Optionally called for some period once the camera stops, so the
        visual can finish up what it was doing, such as loading data into
        VRAM or animating itself.
        """
        pass

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.events, self)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


def _prepare_transform(transform: Affine) -> np.ndarray:
    # convert NumPy axis ordering to VisPy axis ordering
    # by reversing the axes order and flipping the linear
    # matrix
    translate = transform.translate[::-1]
    matrix = transform.linear_matrix[::-1, ::-1].T

    # Embed in the top left corner of a 4x4 affine matrix
    affine_matrix = np.eye(4)
    affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
    affine_matrix[-1, : len(translate)] = translate

    return affine_matrix
