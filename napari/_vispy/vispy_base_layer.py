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

        MAX_TEXTURE_SIZE_2D, MAX_TEXTURE_SIZE_3D = get_max_texture_sizes()
        self.MAX_TEXTURE_SIZE_2D = MAX_TEXTURE_SIZE_2D
        self.MAX_TEXTURE_SIZE_3D = MAX_TEXTURE_SIZE_3D

        self.layer.events.refresh.connect(lambda e: self.node.update())
        self.layer.events.set_data.connect(self._on_data_change)
        self.layer.events.visible.connect(self._on_visible_change)
        self.layer.events.opacity.connect(self._on_opacity_change)
        self.layer.events.blending.connect(self._on_blending_change)
        self.layer.events.scale.connect(self._on_matrix_change)
        self.layer.events.translate.connect(self._on_matrix_change)
        self.layer.events.rotation.connect(self._on_matrix_change)
        self.layer.events.shear.connect(self._on_matrix_change)
        self.layer.events.affine.connect(self._on_matrix_change)

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
        rotation, upper_tri = np.linalg.qr(matrix)
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
            self.layer.dims.displayed
        )
        # convert NumPy axis ordering to VisPy axis ordering
        # using a conjugation operation
        if self.layer.dims.ndisplay == 2:
            t_form = np.array([[0, 1], [-1, 0]])
        else:
            t_form = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        t_form_h = np.eye(t_form.shape[0] + 1)
        t_form_h[:-1, :-1] = t_form
        affine = t_form_h @ transform.affine_matrix @ t_form_h.T
        matrix = affine[:-1, :-1]
        translate = affine[:-1, -1]

        # Embed in 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        if self._array_like:
            # Perform pixel offset
            d2w_matrix = (
                self.layer._transforms['data2world']
                .set_slice(self.layer.dims.displayed)
                .linear_matrix
            )
            d2w_matrix = t_form @ d2w_matrix @ t_form.T
            offset = -d2w_matrix.T @ np.ones(matrix.shape[1]) / 2
            translate += offset
        affine_matrix[-1, : len(translate)] = translate
        self._master_transform.matrix = affine_matrix

    def _reset_base(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()


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
