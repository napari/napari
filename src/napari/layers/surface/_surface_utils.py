import numpy as np


def calculate_barycentric_coordinates(
    point: np.ndarray, triangle_vertices: np.ndarray
) -> np.ndarray:
    """Calculate the barycentric coordinates for a point in a triangle.

    http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates

    Parameters
    ----------
    point : np.ndarray
        The coordinates of the point for which to calculate the barycentric coordinate.
    triangle_vertices : np.ndarray
        (3, D) array containing the triangle vertices.

    Returns
    -------
    barycentric_coorinates : np.ndarray
        The barycentric coordinate [u, v, w], where u, v, and w are the
        barycentric coordinates for the first, second, third triangle
        vertex, respectively.
    """
    vertex_a = triangle_vertices[0, :]
    vertex_b = triangle_vertices[1, :]
    vertex_c = triangle_vertices[2, :]

    v0 = vertex_b - vertex_a
    v1 = vertex_c - vertex_a
    v2 = point - vertex_a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1 - v - w
    return np.array([u, v, w])
