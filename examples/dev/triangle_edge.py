"""Visualize the triangulation algorithms used by the Shapes layer.

This example uses napari layers to draw each of the components of a face and
edge triangulation in a Shapes layer.

Shapes layers don't "just" draw polygons, ellipses, and so on: each shape,
as well as its borders, is broken down into a collection of triangles (this
is called a *triangulation*), which are then sent to OpenGL for drawing:
drawing triangles is one of the "visualization primitives" in OpenGL and most
2D and 3D drawing frameworks.

It turns out that converting arbitrary shapes into a collection of triangles
can be quite tricky: very shallow angles cause errors in the algorithms, and
can also make certain desirable properties (such as edges not overlapping with
each other when a polygon makes a sharp turn) actually impossible to achieve
with fast (single-pass) algorithms.

This example draws the Shapes layer normally, but also overlays all the
elements of the triangulation: the triangles themselves, and the normal vectors
on each polygon vertex, from which the triangulation is computed.
"""

import typing

import numba
import numpy as np

import napari
from napari.layers import Points, Shapes, Vectors
from napari.layers.shapes._shapes_utils import generate_2D_edge_meshes


def generate_regular_polygon(n, radius=1):
    """Generate vertices of a regular n-sided polygon centered at the origin.

    Parameters
    ----------
    n : int
        The number of sides (vertices).
    radius : float, optional
        The radius of the circumscribed circle.

    Returns
    -------
    np.ndarray
        An array of shape (n, 2) containing the vertex coordinates.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def generate_order_vectors(path_, closed) -> np.ndarray:
    """Generate the vectors tangent to the path.

    Parameters
    ----------
    path_ : np.ndarray, shape (N, 2)
        A list of 2D path coordinates.
    closed : bool
        Whether the coordinates represent a closed polygon or an open
        line/path.

    Returns
    -------
    vec : np.ndarray, shape (N, 2, 2)
        A set of vectors, defined by a 2D position and a 2D projection.
    """
    raw_vecs = np.diff(path_, axis=0)
    norm = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    normals = raw_vecs / norm
    vec = np.empty((path_.shape[0], 2, 2))
    vec[:, 0] = path_
    vec[:-1, 1] = normals
    if closed:
        vec[-1, 1] = (
                (path_[0] - path_[-1]) / np.linalg.norm(path_[-1] - path_[0])
                )
    else:
        vec[-1, 1] = -vec[-2, 1]
    return vec


def generate_miter_helper_vectors(normal_vector_arr: np.ndarray) -> np.ndarray:
    """Generate the miter helper vectors.

    For each point on the path, the miter helper vectors are pairs of vectors
    First vector is the normal vector scaled by 1/2
    Second vector is the normal vector of previous point scaled by -1/2

    https://github.com/napari/napari/pull/7268#user-content-miter
    Blue vectors on the image.

    Parameters
    ----------
    normal_vector_arr : array of shape (n, 2)
        array of normal vectors representing edges in the path

    Returns
    -------
    array of shape (n, 2, 2)
        array of miter helper vectors
    """
    half_of_normal = normal_vector_arr.copy()
    half_of_normal[:, 1] *= 0.5
    half_of_prev_normal = half_of_normal.copy()
    half_of_prev_normal[:, 1] *= -1
    half_of_prev_normal[:, 1] = np.roll(half_of_prev_normal[:, 1], 1, axis=0)
    half_of_prev_normal[:, 0] += half_of_normal[:, 1]
    return np.concatenate([half_of_normal, half_of_prev_normal], axis=0)

@numba.njit
def generate_orthogonal_vectors(normal_vector: np.ndarray) -> np.ndarray:
    """Generate the orthogonal vectors to the normal vectors.

    The orthogonal vector starts at the middle of the normal vector and is orthogonal to it.
    It has length equal to half of the normal vector.
    """
    vec1 = normal_vector.copy()
    vec1[:, 1] *= 0.5
    orthogonal_vector = vec1.copy()

    orthogonal_vector[:, 0] += vec1[:, 1]
    orthogonal_vector[:, 1, 0] = -vec1[:, 1, 1]
    orthogonal_vector[:, 1, 1] = vec1[:, 1, 0]
    return orthogonal_vector

@numba.njit
def generate_mitter_vectors(mesh) -> np.ndarray:
    """For each point on path, generate the vectors pointing to adjusted points. Generate the mitter vectors for each point on the path.
    """
    vec_points = np.empty((mesh[0].shape[0], 2, 2))
    vec_points[:, 0] = mesh[0]
    vec_points[:, 1] = mesh[1]
    return vec_points

@numba.njit
def generate_edge_triangle_borders(centers, offsets, triangles) -> np.ndarray:
    """For each triangle in mesh generate 3 vectors that represent the borders of the triangle.
    """
    borders = np.empty((len(triangles)*3, 2, 2), dtype=centers.dtype)
    for i, triangle in enumerate(triangles):
        a, b, c = triangle
        a1 = centers[a] + offsets[a]
        b1 = centers[b] + offsets[b]
        c1 = centers[c] + offsets[c]
        borders[i * 3, 0] = a1
        borders[i * 3, 1] = (b1 - a1)
        borders[i * 3 + 1, 0] = b1
        borders[i * 3 + 1, 1] = (c1 - b1)
        borders[i * 3 + 2, 0] = c1
        borders[i * 3 + 2, 1] = (a1 - c1)
    return borders

@numba.njit
def generate_face_triangle_borders(vertices, triangles) -> np.ndarray:
    """For each triangle in mesh generate 3 vectors that represent the borders of the triangle.
    """
    borders = np.empty((len(triangles)*3, 2, 2), dtype=vertices.dtype)
    for i, triangle in enumerate(triangles):
        a, b, c = triangle
        a1 = vertices[a]
        b1 = vertices[b]
        c1 = vertices[c]
        borders[i * 3, 0] = a1
        borders[i * 3, 1] = (b1 - a1)
        borders[i * 3 + 1, 0] = b1
        borders[i * 3 + 1, 1] = (c1 - b1)
        borders[i * 3 + 2, 0] = c1
        borders[i * 3 + 2, 1] = (a1 - c1)
    return borders


path = np.array([[0,0], [0,1], [1,1], [1,0]]) * 10

sharp = np.array([[1, 1], [10, 0], [1, -1], [0, -10], [-1, -1], [-10, 0], [-1, 1], [0, 10]])
sharp2 = np.array([[2, 10], [0, -5], [-2, 10], [-2, -10], [2, -10]])

polygons = [
    generate_regular_polygon(4, radius=1) * 10,
    generate_regular_polygon(10, radius=1) * 10 + np.array([[25, 0]]),
    generate_regular_polygon(3, radius=1) * 10 + np.array([[0, 25]]),
    sharp2 + np.array([[25, 25]]),
    sharp + np.array([[50, 0]]),
    sharp[::-1] + np.array([[50, 26]]),
    path + np.array([[0, 50]]),
    generate_regular_polygon(10, radius=1) * 10 + np.array([[25, 50]]),
    np.array([[0, -10], [0, 0], [0, 10]]) + np.array([[50, 50]]),
    np.array(
    [
        [10.97627008, 14.30378733],
        [12.05526752, 10.89766366],
        [8.47309599, 12.91788226],
        [8.75174423, 17.83546002],
        [19.27325521, 7.66883038],
        [15.83450076, 10.5778984],
    ]
    ) + np.array([[60, -15]]),
]

shape_type=['polygon'] * 6 + ['path'] * 3 + ['polygon']
s = Shapes(polygons, shape_type=shape_type, name="shapes")


mesh1_li = [generate_2D_edge_meshes(p, closed=s != 'path') for p, s in zip(polygons, shape_type)]

mesh1 = tuple(np.concatenate(el, axis=0) for el in zip(*mesh1_li))


class Helpers(typing.NamedTuple):
    points: np.ndarray
    order_vectors: np.ndarray
    miter_helper: np.ndarray
    orthogonal_vector: np.ndarray
    miter_vectors: np.ndarray
    triangles_vectors: np.ndarray
    face_triangles_vectors: np.ndarray

def get_helper_data_from_shapes(shape: Shapes) -> Helpers:
    shapes = shape._data_view.shapes
    mesh_list = [(x._edge_vertices, x._edge_offsets, x._edge_triangles) for x in shapes]
    path_list = [(x.data, x._closed) for x in shapes]
    mesh = tuple(np.concatenate(el, axis=0) for el in zip(*mesh_list))
    face_mesh_list = [(x._face_vertices, x._face_triangles) for x in shapes if len(x._face_vertices)]

    points = mesh[0] + mesh[1]
    order_vectors_li = [generate_order_vectors(path_, closed) for path_, closed in path_list]
    order_vectors = np.concatenate(order_vectors_li, axis=0)
    miter_helper = np.concatenate([generate_miter_helper_vectors(o) for o in order_vectors_li], axis=0)
    orthogonal_vector = np.concatenate([generate_orthogonal_vectors(o) for o in order_vectors_li], axis=0)
    miter_vectors = np.concatenate([generate_mitter_vectors(m) for m in mesh_list], axis=0)
    triangles_vectors = np.concatenate([generate_edge_triangle_borders(*m) for m in mesh_list], axis=0)
    face_triangles_vectors = np.concatenate([generate_face_triangle_borders(*m) for m in face_mesh_list], axis=0)

    return Helpers(
        points=points,
        order_vectors=order_vectors,
        miter_helper=miter_helper,
        orthogonal_vector=orthogonal_vector,
        miter_vectors=miter_vectors,
        triangles_vectors=triangles_vectors,
        face_triangles_vectors=face_triangles_vectors,
    )


def update_layers():
    shapes_layer = v.layers['shapes']

    helpers = get_helper_data_from_shapes(shapes_layer)
    v.layers['join points'].data = helpers.points
    v.layers['normal vectors'].data = helpers.order_vectors
    v.layers['miter helper'].data = helpers.miter_helper
    v.layers['orthogonal'].data = helpers.orthogonal_vector
    v.layers['miter vectors'].data = helpers.miter_vectors
    v.layers['triangle face vectors'].data = helpers.face_triangles_vectors
    v.layers['triangle vectors'].data = helpers.triangles_vectors


def add_helper_layers(viewer: napari.Viewer):
    shapes = viewer.layers['shapes']
    helpers = get_helper_data_from_shapes(shapes)
    p = Points(helpers.points, size=0.1, face_color='white', name='join points')
    ve = Vectors(helpers.order_vectors, edge_width=0.1, vector_style='arrow', name='normal vectors')
    ve2 = Vectors(helpers.miter_helper, edge_width=0.06, vector_style='arrow', edge_color="blue", name="miter helper")
    ve3 = Vectors(helpers.orthogonal_vector, edge_width=0.04, vector_style='arrow', edge_color="green", name='orthogonal')
    ve4 = Vectors(helpers.miter_vectors, edge_width=0.05, vector_style='arrow', edge_color="yellow", name='miter vectors')
    ve5 = Vectors(helpers.face_triangles_vectors, edge_width=0.04, vector_style='arrow', edge_color="magenta", name='triangle face vectors')
    ve6 = Vectors(helpers.triangles_vectors, edge_width=0.04, vector_style='arrow', edge_color="pink", name='triangle vectors')
    viewer.add_layer(p)
    viewer.add_layer(ve)
    viewer.add_layer(ve2)
    viewer.add_layer(ve3)
    viewer.add_layer(ve4)
    viewer.add_layer(ve5)
    viewer.add_layer(ve6)


def get_nono_accelerated_points(shape: Shapes) -> np.ndarray:
    """Get the non-accelerated points"""
    shapes = shape._data_view.shapes
    path_list = [(x.data, x._closed) for x in shapes]
    mesh_list = [generate_2D_edge_meshes(path, closed) for path, closed in path_list]
    mesh = tuple(np.concatenate(el, axis=0) for el in zip(*mesh_list))

    return mesh[0] + mesh[1]


def update_non_accelerated_points():
    data = get_nono_accelerated_points(v.layers["shapes"])
    v.layers["non accelerated join points"].data = data



def add_non_accelerated_points(viewer: napari.Viewer):
    data = get_nono_accelerated_points(v.layers["shapes"])
    p = Points(data, size=0.2, face_color='yellow', name='non accelerated join points')
    viewer.add_layer(p)


v = napari.Viewer()
v.add_layer(s)

add_non_accelerated_points(v)
add_helper_layers(v)


s.events.set_data.connect(update_layers)
s.events.set_data.connect(update_non_accelerated_points)


v.camera.center = (0, 25, 25)
v.camera.zoom = 50

napari.run()
