import numpy as np

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.visuals.vectors import VectorsVisual
from napari.layers.utils.layer_utils import segment_normal


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer) -> None:
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
            self.layer.vector_style,
        )
        face_color = self.layer._view_face_color
        ndisplay = self.layer._slice_input.ndisplay
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


def generate_vector_meshes(vectors, width, length, vector_style):
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
    vertices : (4N, 2) array for 2D and (8N, 2) array for 3D
        Vertices of all triangles for the lines
    triangles : (2N, 3) array for 2D or (4N, 3) array for 3D
        Vertex indices that form the mesh triangles
    """
    ndim = vectors.shape[2]
    if ndim == 2:
        vertices, triangles = generate_vector_meshes_2D(
            vectors, width, length, vector_style
        )
    else:
        v_a, t_a = generate_vector_meshes_2D(
            vectors, width, length, vector_style, p=(0, 0, 1)
        )
        v_b, t_b = generate_vector_meshes_2D(
            vectors, width, length, vector_style, p=(1, 0, 0)
        )
        vertices = np.concatenate([v_a, v_b], axis=0)
        triangles = np.concatenate([t_a, len(v_a) + t_b], axis=0)

    return vertices, triangles


def generate_vector_meshes_2D(
    vectors, width, length, vector_style, p=(0, 0, 1)
):
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
    vector_style : str
        display style of the vectors
    p : 3-tuple, optional
        orthogonal vector for segment calculation in 3D.

    Returns
    -------
    vertices : (4N, D) array
        Vertices of all triangles for the lines
    triangles : (2N, 3) array
        Vertex indices that form the mesh triangles
    """

    if vector_style == 'line':
        vertices, triangles = generate_meshes_line_2D(
            vectors, width, length, p
        )

    elif vector_style == 'triangle':
        vertices, triangles = generate_meshes_triangle_2D(
            vectors, width, length, p
        )

    elif vector_style == 'arrow':
        vertices, triangles = generate_meshes_arrow_2D(
            vectors, width, length, p
        )

    else:
        raise NotImplementedError

    return vertices, triangles


def generate_meshes_line_2D(vectors, width, length, p):
    """Generates list of mesh vertices and triangles from a list of vectors.

    Vectors are composed of 4 vertices and 2 triangles.
    Vertices are generated according to the following scheme:
    1---x---0
    | .     |
    |   .   |
    |     . |
    3---v---2

    Where x marks the start point of the vector, and v its end point.

    In the case of k 2D vectors, the output 'triangles' is:
    [
        [0,1,2],                # vector 0,   triangle i=0
        [1,2,3],                # vector 0,   triangle i=1
        [4,5,6],                # vector 1,   triangle i=2
        [5,6,7],                # vector 1,   triangle i=3

        ...,

        [2i, 2i + 1, 2i + 2],   # vector k-1, triangle i=2k-2 (i%2=0)
        [2i - 1, 2i, 2i + 1]    # vector k-1, triangle i=2k-1 (i%2=1)
    ]
    """
    nvectors, _, ndim = vectors.shape

    vectors_starts = vectors[:, 0]
    vectors_ends = vectors_starts + length * vectors[:, 1]

    vertices = np.zeros((4 * nvectors, ndim))
    offsets = segment_normal(vectors_starts, vectors_ends, p=p)
    offsets = np.repeat(offsets, 4, axis=0)

    signs = np.ones((len(offsets), ndim))
    signs[::2] = -1
    offsets = offsets * signs

    vertices[::4] = vectors_starts
    vertices[1::4] = vectors_starts
    vertices[2::4] = vectors_ends
    vertices[3::4] = vectors_ends

    vertices = vertices + width * offsets / 2

    triangles = np.array(
        [
            [2 * i, 2 * i + 1, 2 * i + 2]
            if i % 2 == 0
            else [2 * i - 1, 2 * i, 2 * i + 1]
            for i in range(2 * nvectors)
        ]
    ).astype(np.uint32)

    return vertices, triangles


def generate_meshes_triangle_2D(vectors, width, length, p):
    """Generates list of mesh vertices and triangles from a list of vectors.

    Vectors are composed of 3 vertices and 1 triangles.
    Vertices are generated according to the following scheme:
    1---x---0
     .     .
      .   .
       . .
        2


    Where x marks the start point of the vector, and the vertex 2 its end
    point.

    In the case of k 2D vectors, the output 'triangles' is:
    [
        [0,1,2],                # vector 0,   triangle i=0
        [3,4,5],                # vector 1,   triangle i=1

        ...,

        [3i, 3i + 1, 3i + 2]    # vector k-1, triangle i=k-1
    ]
    """
    nvectors, _, ndim = vectors.shape

    vectors_starts = vectors[:, 0]
    vectors_ends = vectors_starts + length * vectors[:, 1]

    vertices = np.zeros((3 * nvectors, ndim))
    offsets = segment_normal(vectors_starts, vectors_ends, p=p)
    offsets = np.repeat(offsets, 3, axis=0)

    signs = np.ones((len(offsets), ndim))
    signs[::3] = -1
    multipliers = np.ones((len(offsets), ndim))
    multipliers[2::3] = 0
    offsets = offsets * signs * multipliers

    vertices[::3] = vectors_starts
    vertices[1::3] = vectors_starts
    vertices[2::3] = vectors_ends

    vertices = vertices + width * offsets / 2

    # faster than using the formula in the docstring
    triangles = np.arange(3 * nvectors).reshape((-1, 3)).astype(np.uint32)

    return vertices, triangles


def generate_meshes_arrow_2D(vectors, width, length, p):
    """Generates list of mesh vertices and triangles from a list of vectors.

    Vectors are composed of 7 vertices and 3 triangles.
    Vertices are generated according to the following scheme:
        1---x---0
        | .     |
        |   .   |
        |     . |
    5---3-------2---4
       .         .
          .   .
            6

    Where x marks the start point of the vector, and the vertex 6 its end
    point.

    In the case of k 2D vectors, the output 'triangles' is:
    [
        [0,1,2],                # vector 0,   triangle i=0
        [1,2,3],                # vector 0,   triangle i=1
        [4,5,6],                # vector 0,   triangle i=2
        [7,8,9],                # vector 1,   triangle i=3
        [8,9,10],               # vector 1,   triangle i=4
        [11,12,13],             # vector 1,   triangle i=5

        ...,

        [7i/3,           7i/3 + 1,       7i/3 + 2],
            # vector k-1, triangle i=3k-3 (i%3=0)
        [7(i - 1)/3 + 1, 7(i - 1)/3 + 2, 7(i - 1)/3 + 3],
            # vector k-1, triangle i=3k-2 (i%3=1)
        [7(i - 2)/3 + 4, 7(i - 2)/3 + 5, 7(i - 2)/3 + 6]
            # vector k-1, triangle i=3k-1 (i%3=2)
    ]
    """
    nvectors, _, ndim = vectors.shape

    vectors_starts = vectors[:, 0]

    # Will be used to generate the vertices 2,3,4 and 5.
    # Right now the head of the arrow is put at 75% of the length
    # of the vector.
    vectors_intermediates = vectors_starts + 0.75 * vectors[:, 1]

    vectors_ends = vectors_starts + length * vectors[:, 1]

    vertices = np.zeros((7 * nvectors, ndim))
    offsets = segment_normal(vectors_starts, vectors_ends, p=p)
    offsets = np.repeat(offsets, 7, axis=0)

    signs = np.ones((len(offsets), ndim))
    signs[::2] = -1
    multipliers = np.ones((len(offsets), ndim))
    multipliers[4::7] = 2
    multipliers[5::7] = 2
    multipliers[6::7] = 0
    offsets = offsets * signs * multipliers

    vertices[::7] = vectors_starts
    vertices[1::7] = vectors_starts
    vertices[2::7] = vectors_intermediates
    vertices[3::7] = vectors_intermediates
    vertices[4::7] = vectors_intermediates
    vertices[5::7] = vectors_intermediates
    vertices[6::7] = vectors_ends

    vertices = vertices + width * offsets / 2

    triangles = np.array(
        [
            [7 * i / 3, 7 * i / 3 + 1, 7 * i / 3 + 2]
            if i % 3 == 0
            else [
                7 * (i - 1) / 3 + 1,
                7 * (i - 1) / 3 + 2,
                7 * (i - 1) / 3 + 3,
            ]
            if i % 3 == 1
            else [
                7 * (i - 2) / 3 + 4,
                7 * (i - 2) / 3 + 5,
                7 * (i - 2) / 3 + 6,
            ]
            for i in range(3 * nvectors)
        ]
    ).astype(np.uint32)

    return vertices, triangles
