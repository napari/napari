import numpy as np
import pytest

from napari._vispy.layers.vectors import (
    generate_vector_meshes,
    generate_vector_meshes_2D,
)


@pytest.mark.parametrize(
    "edge_width, length, dims, style",
    [
        [0, 0, 2, 'line'],
        [0.3, 0.3, 2, 'line'],
        [1, 1, 3, 'line'],
        [0, 0, 2, 'triangle'],
        [0.3, 0.3, 2, 'triangle'],
        [1, 1, 3, 'triangle'],
        [0, 0, 2, 'arrow'],
        [0.3, 0.3, 2, 'arrow'],
        [1, 1, 3, 'arrow'],
    ],
)
def test_generate_vector_meshes(edge_width, length, dims, style):
    n = 10

    data = np.random.random((n, 2, dims))
    vertices, faces = generate_vector_meshes(
        data, width=edge_width, length=length, vector_style=style
    )
    vertices_length, vertices_dims = vertices.shape
    faces_length, faces_dims = faces.shape

    if dims == 2:
        if style == 'line':
            assert vertices_length == 4 * n
            assert faces_length == 2 * n
        elif style == 'triangle':
            assert vertices_length == 3 * n
            assert faces_length == n
        elif style == 'arrow':
            assert vertices_length == 7 * n
            assert faces_length == 3 * n

    elif dims == 3:
        if style == 'line':
            assert vertices_length == 8 * n
            assert faces_length == 4 * n
        elif style == 'triangle':
            assert vertices_length == 6 * n
            assert faces_length == 2 * n
        elif style == 'arrow':
            assert vertices_length == 14 * n
            assert faces_length == 6 * n

    assert vertices_dims == dims
    assert faces_dims == 3


@pytest.mark.parametrize(
    "edge_width, length, style, p",
    [
        [0, 0, 'line', (1, 0, 0)],
        [0.3, 0.3, 'line', (0, 1, 0)],
        [1, 1, 'line', (0, 0, 1)],
        [0, 0, 'triangle', (1, 0, 0)],
        [0.3, 0.3, 'triangle', (0, 1, 0)],
        [1, 1, 'triangle', (0, 0, 1)],
        [0, 0, 'arrow', (1, 0, 0)],
        [0.3, 0.3, 'arrow', (0, 1, 0)],
        [1, 1, 'arrow', (0, 0, 1)],
    ],
)
def test_generate_vector_meshes_2D(edge_width, length, style, p):
    n = 10
    dims = 2

    data = np.random.random((n, 2, dims))
    vertices, faces = generate_vector_meshes_2D(
        data, width=edge_width, length=length, vector_style=style, p=p
    )
    vertices_length, vertices_dims = vertices.shape
    faces_length, faces_dims = faces.shape

    if style == 'line':
        assert vertices_length == 4 * n
        assert faces_length == 2 * n
    elif style == 'triangle':
        assert vertices_length == 3 * n
        assert faces_length == n
    elif style == 'arrow':
        assert vertices_length == 7 * n
        assert faces_length == 3 * n

    assert vertices_dims == dims
    assert faces_dims == 3


@pytest.mark.parametrize(
    "initial_vector_style, new_vector_style",
    [
        ['line', 'line'],
        ['line', 'triangle'],
        ['line', 'arrow'],
        ['triangle', 'line'],
        ['triangle', 'triangle'],
        ['triangle', 'arrow'],
        ['arrow', 'line'],
        ['arrow', 'triangle'],
        ['arrow', 'arrow'],
    ],
)
def test_vector_style_change(
    make_napari_viewer, initial_vector_style, new_vector_style
):
    # initialize viewer
    viewer = make_napari_viewer()
    # add a vector layer
    vector_layer = viewer.add_vectors(
        vector_style=initial_vector_style, name='vectors'
    )

    class Counter:
        def __init__(self):
            self.count = 0

        def increment_count(self, event):
            self.count += 1

    # initialize counter
    counter = Counter()
    # connect counter to vector_style change
    vector_layer.events.vector_style.connect(counter.increment_count)

    # change vector_style
    vector_layer.vector_style = new_vector_style

    # check if counter was called
    if initial_vector_style == new_vector_style:
        assert counter.count == 0
    else:
        assert counter.count == 1
