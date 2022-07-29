from copy import copy

import numpy as np

from ..visuals.vectors import VectorsVisual
from .base import VispyBaseLayer


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = VectorsVisual()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(self._on_data_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self):

        # Make meshes
        vertices, faces = generate_vector_meshes(
            self.layer._view_data,
            self.layer.edge_width,
            self.layer.length,
        )
        face_color = self.layer._view_face_color
        ndisplay = self.layer._ndisplay
        ndim = self.layer.ndim

        if len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, ndisplay))
            faces = np.array([[0, 1, 2]])
            face_color = np.array([[0, 0, 0, 0]])
        else:
            vertices = vertices[:, ::-1]

        if ndisplay == 3 and ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        self.node.set_data(
            vertices=vertices,
            faces=faces,
            face_colors=face_color,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()


def generate_vector_meshes(vectors, width, length):
    """Generates list of mesh vertices and triangles from a list of vectors

    Parameters
    ----------
    vectors : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions, where D is 2 or 3.
    width : float
        width of the line to be drawn
    length : float
        length multiplier of the line to be drawn

    Returns
    -------
    vertices : (4N, D) array
        Vertices of all triangles for the lines
    triangles : (2N, 3) array
        Vertex indices that form the mesh triangles
    """
    ndim = vectors.shape[2]
    if ndim == 2:
        vertices, triangles = generate_vector_meshes_2D(vectors, width, length)
    else:
        v_a, t_a = generate_vector_meshes_2D(
            vectors, width, length, p=(0, 0, 1)
        )
        v_b, t_b = generate_vector_meshes_2D(
            vectors, width, length, p=(1, 0, 0)
        )
        vertices = np.concatenate([v_a, v_b], axis=0)
        triangles = np.concatenate([t_a, len(v_a) + t_b], axis=0)

    return vertices, triangles


def generate_vector_meshes_2D(vectors, width, length, p=(0, 0, 1)):
    """Generates list of mesh vertices and triangles from a list of vectors

    Parameters
    ----------
    vectors : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions, where D is 2 or 3.
    width : float
        width of the line to be drawn
    length : float
        length multiplier of the line to be drawn
    p : 3-tuple, optional
        orthogonal vector for segment calculation in 3D.

    Returns
    -------
    vertices : (4N, D) array
        Vertices of all triangles for the lines
    triangles : (2N, 3) array
        Vertex indices that form the mesh triangles
    """
    ndim = vectors.shape[2]
    vectors = np.reshape(copy(vectors), (-1, ndim))
    vectors[1::2] = vectors[::2] + length * vectors[1::2]

    centers = np.repeat(vectors, 2, axis=0)
    offsets = segment_normal(vectors[::2, :], vectors[1::2, :], p=p)
    offsets = np.repeat(offsets, 4, axis=0)
    signs = np.ones((len(offsets), ndim))
    signs[::2] = -1
    offsets = offsets * signs

    vertices = centers + width * offsets / 2
    triangles = np.array(
        [
            [2 * i, 2 * i + 1, 2 * i + 2]
            if i % 2 == 0
            else [2 * i - 1, 2 * i, 2 * i + 1]
            for i in range(len(vectors))
        ]
    ).astype(np.uint32)

    return vertices, triangles


def segment_normal(a, b, p=(0, 0, 1)):
    """Determines the unit normal of the vector from a to b.

    Parameters
    ----------
    a : np.ndarray
        Length 2 array of first point or Nx2 array of points
    b : np.ndarray
        Length 2 array of second point or Nx2 array of points
    p : 3-tuple, optional
        orthogonal vector for segment calculation in 3D.

    Returns
    -------
    unit_norm : np.ndarray
        Length the unit normal of the vector from a to b. If a == b,
        then returns [0, 0] or Nx2 array of vectors
    """
    d = b - a

    if d.ndim == 1:
        if len(d) == 2:
            normal = np.array([d[1], -d[0]])
        else:
            normal = np.cross(d, p)
        norm = np.linalg.norm(normal)
        if norm == 0:
            norm = 1
    else:
        if d.shape[1] == 2:
            normal = np.stack([d[:, 1], -d[:, 0]], axis=0).transpose(1, 0)
        else:
            normal = np.cross(d, p)

        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        ind = norm == 0
        norm[ind] = 1
    unit_norm = normal / norm

    return unit_norm
