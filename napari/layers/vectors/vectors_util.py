import numpy as np
from copy import copy
from ...util import segment_normal


def vectors_to_coordinates(vectors):
    """Validate and convert vector data to a coordinate representation

    Parameters
    ----------
    vectors : (N, 2, D) or (N1, N2, ..., ND, D) array
        A (N, 2, D) array is interpreted as "coordinate-like" data and a list
        of N vectors with start point and projections of the vector in D
        dimensions. A (N1, N2, ..., ND, D) array is interpreted as
        "image-like" data where there is a length D vector of the
        projections at each pixel.

    Returns
    ----------
    coords : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions.
    """
    if vectors.shape[-2] == 2 and vectors.ndim == 3:
        # an (N, 2, D) array that is coordinate-like
        coords = vectors
    elif vectors.shape[-1] == vectors.ndim - 1:
        # an (N1, N2, ..., ND, D) array that is image-like
        coords = convert_image_to_coordinates(vectors)
    else:
        raise TypeError(
            "Vector data of shape %s is not supported" % str(vectors.shape)
        )

    return coords


def convert_image_to_coordinates(vectors):
    """To convert an image-like array with elements (y-proj, x-proj) into a
    position list of coordinates
    Every pixel position (n, m) results in two output coordinates of (N,2)

    Parameters
    ----------
    vectors : (N1, N2, ..., ND, D) array
        "image-like" data where there is a length D vector of the
        projections at each pixel.

    Returns
    ----------
    coords : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions.
    """
    # create coordinate spacing for image
    spacing = [list(range(r)) for r in vectors.shape[:-1]]
    grid = np.meshgrid(*spacing)

    # create empty vector of necessary shape
    nvect = np.prod(vectors.shape[:-1])
    coords = np.empty((nvect, 2, vectors.ndim - 1), dtype=np.float32)

    # assign coordinates to all pixels
    for i, g in enumerate(grid):
        coords[:, 0, i] = g.flatten()
    coords[:, 1, :] = np.reshape(vectors, (-1, vectors.ndim - 1))

    return coords


def generate_vector_meshes(vectors, width, length):
    """Generates list of mesh vertices and triangles from a list of vectors

    Parameters
    ----------
    vectors : (N, 2, D) array
        A list of N vectors with start point and projections of the vector
        in D dimensions. Vectors are projected onto the last two
        dimensions if D > 2.
    width : float
        width of the line to be drawn
    length : float
        length multiplier of the line to be drawn

    Returns
    ----------
    vertices : (4N, 2) array
        Vertices of all triangles for the lines
    triangles : (2N, 2) array
        Vertex indices that form the mesh triangles
    """
    vectors = np.reshape(copy(vectors[:, :, -2:]), (-1, 2))
    vectors[1::2] = vectors[::2] + length * vectors[1::2]
    centers = np.repeat(vectors, 2, axis=0)
    offsets = segment_normal(vectors[::2, :], vectors[1::2, :])
    offsets = np.repeat(offsets, 4, axis=0)
    signs = np.ones((len(offsets), 2))
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
