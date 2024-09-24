import numpy as np

import napari
from napari.layers import Points, Shapes, Vectors
from napari.layers.shapes._shapes_utils import (
    acc_generate_2D_edge_meshes,
    generate_2D_edge_meshes,
)


def generate_regular_polygon(n, radius=1):
    """
    Generates the vertices of a regular n-sided polygon centered at the origin.

    Parameters:
    n (int): The number of sides (vertices).
    radius (float): The radius of the circumscribed circle. Default is 1.

    Returns:
    np.ndarray: An array of shape (n, 2) containing the vertices coordinates.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def generate_order_vectors(path_, closed) -> np.ndarray:
    """Generate the vectors tangent to the path."""
    n = path_[1:] - path_[:-1]
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    normals = n / norm
    vec = np.empty((path_.shape[0], 2, 2))
    vec[:, 0] = path_
    vec[:-1, 1] = normals
    if closed:
        vec[-1, 1] = (path_[0] - path_[-1]) / np.linalg.norm(path_[-1] - path_[0])
    else:
        vec[-1, 1] = -vec[-2, 1]
    return vec

def generate_miter_helper_vectors(normal_vector: np.ndarray) -> np.ndarray:
    """Generate the miter helper vectors.

    for each point on the path, the miter helper vectors are pairs of vectors
    First vector is the normal vector scaled by 1/2
    Second vector is the normal vector of previous point scaled by -1/2
    """
    vec2 = normal_vector.copy()
    vec2[:, 1] *= 0.5
    vec3 = vec2.copy()
    vec3[:, 1] *= -1
    vec3[:, 1] = np.roll(vec3[:, 1], 1, axis=0)
    vec3[:, 0] += vec2[:, 1]
    return np.concatenate([vec2, vec3], axis=0)

def generate_orthogonal_vectors(normal_vector: np.ndarray) -> np.ndarray:
    """Generate the orthogonal vectors to the normal vectors.

    The orthogonal vector starts at the middle of the normal vector and is orthogonal to it.
    It has length equal to half of the normal vector.
    """
    vec2 = normal_vector.copy()
    vec2[:, 1] *= 0.5
    vec5 = vec2.copy()
    vec5[:, 0] += vec2[:, 1]
    vec5[:, 1, 0] = -vec2[:, 1, 1]
    vec5[:, 1, 1] = vec2[:, 1, 0]
    return vec5

def generate_mitter_vectors(mesh) -> np.ndarray:
    """For each point on path, generate the vectors pointing to adjusted points. Generate the mitter vectors for each point on the path.
    """
    vec_points = np.empty((mesh[0].shape[0], 2, 2))
    vec_points[:, 0] = mesh[0]
    vec_points[:, 1] = mesh[1]
    return vec_points

def generate_triangle_borders(mesh) -> np.ndarray:
    """For each triangle in mesh generate 3 vectors that represent the borders of the triangle.
    """
    res = []
    centers, offsets, triangles = mesh
    for triangle in triangles:
        a, b, c = triangle
        a1 = centers[a] + offsets[a]
        b1 = centers[b] + offsets[b]
        c1 = centers[c] + offsets[c]
        res.extend([[a1, (b1 - a1)], [b1, (c1 - b1)], [c1, (a1 - c1)]])
    return np.array(res)

path = np.array([[0,0], [0,1], [1,1], [1,0]]) * 10
# path = generate_regular_polygon(4, radius=1) * 10

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
s = Shapes(polygons, shape_type=shape_type)


mesh1_li = [generate_2D_edge_meshes(p, closed=s != 'path') for p, s in zip(polygons, shape_type)]
mesh2_li = [
    acc_generate_2D_edge_meshes(p, closed=s != 'path') for p, s in zip(polygons, shape_type)
]

mesh1 = tuple(np.concatenate(el, axis=0) for el in zip(*mesh1_li))
mesh2 = tuple(np.concatenate(el, axis=0) for el in zip(*mesh2_li))


order_vectors_li = [generate_order_vectors(p, s!='path') for p, s in zip(polygons, shape_type)]

order_vectors = np.concatenate(order_vectors_li, axis=0)

miter_helper = np.concatenate([generate_miter_helper_vectors(o) for o in order_vectors_li], axis=0)

orthogonal_vector = np.concatenate([generate_orthogonal_vectors(o) for o in order_vectors_li], axis=0)

miter_vectors = np.concatenate([generate_mitter_vectors(m) for m in mesh2_li], axis=0)

triangles_vectors = np.concatenate([generate_triangle_borders(m) for m in mesh2_li], axis=0)


s = Shapes(polygons, shape_type=shape_type)
p = Points(mesh2[0] + mesh2[1], size=0.2, face_color='white', name='new')
p1 = Points(mesh1[0] + mesh1[1], size=0.1, face_color='yellow', name='original')
ve = Vectors(order_vectors, edge_width=0.1, vector_style='arrow', name='order')
ve2 = Vectors(miter_helper, edge_width=0.06, vector_style='arrow', edge_color="blue")
ve3 = Vectors(orthogonal_vector, edge_width=0.04, vector_style='arrow', edge_color="green", name='orthogonal')
ve4 = Vectors(miter_vectors, edge_width=0.05, vector_style='arrow', edge_color="yellow", name='miter vectors')
ve5 = Vectors(triangles_vectors, edge_width=0.04, vector_style='arrow', edge_color="pink", name='triangle vectors')

v = napari.Viewer()
v.add_layer(s)
v.add_layer(p)
v.add_layer(p1)
v.add_layer(ve)
v.add_layer(ve2)
v.add_layer(ve3)
v.add_layer(ve4)
v.add_layer(ve5)
v.camera.center = (0, 75, 0)
v.camera.zoom = 50

v.window._qt_viewer.console.push({"mesh2_li": mesh2_li})

napari.run()
