import numpy as np
import pytest

from napari._tests.utils import check_layer_world_data_extent
from napari.layers import Surface


def test_random_surface():
    """Test instantiating Surface layer with random 2D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 2
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert np.all(layer.vertices == vertices)
    assert np.all(layer.faces == faces)
    assert np.all(layer.vertex_values == values)
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1


def test_random_surface_no_values():
    """Test instantiating Surface layer with random 2D data but no vertex values."""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    data = (vertices, faces)
    layer = Surface(data)
    assert layer.ndim == 2
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert np.all(layer.vertices == vertices)
    assert np.all(layer.faces == faces)
    assert np.all(layer.vertex_values == np.ones(len(vertices)))
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1


def test_random_3D_surface():
    """Test instantiating Surface layer with random 3D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 3
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(ndisplay=3)
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_random_4D_surface():
    """Test instantiating Surface layer with random 4D data."""
    np.random.seed(0)
    vertices = np.random.random((10, 4))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 4
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(ndisplay=3)
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_random_3D_timeseries_surface():
    """Test instantiating Surface layer with random 3D timeseries data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random((22, 10))
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 4
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1
    assert layer.extent.data[1][0] == 22

    layer._slice_dims(ndisplay=3)
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1

    # If a values axis is made to be a displayed axis then no data should be
    # shown
    with pytest.warns(UserWarning):
        layer._slice_dims(ndisplay=3, order=[3, 0, 1, 2])
        assert len(layer._data_view) == 0


def test_random_3D_multitimeseries_surface():
    """Test instantiating Surface layer with random 3D multitimeseries data."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random((16, 22, 10))
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.ndim == 5
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1
    assert layer.extent.data[1][0] == 16
    assert layer.extent.data[1][1] == 22

    layer._slice_dims(ndisplay=3)
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_changing_surface():
    """Test changing surface layer data"""
    np.random.seed(0)
    vertices = np.random.random((10, 2))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)

    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer.data = data
    assert layer.ndim == 3
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer._data_view.shape[1] == 2
    assert layer._view_vertex_values.ndim == 1

    layer._slice_dims(ndisplay=3)
    assert layer._data_view.shape[1] == 3
    assert layer._view_vertex_values.ndim == 1


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Surface(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_surface_gamma():
    """Test setting gamma."""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)
    assert layer.gamma == 1

    # Change gamma property
    gamma = 0.7
    layer.gamma = gamma
    assert layer.gamma == gamma

    # Set gamma as keyword argument
    layer = Surface(data, gamma=gamma)
    assert layer.gamma == gamma


def test_world_data_extent():
    """Test extent after applying transforms."""
    data = [(-5, 0), (0, 15), (30, 12)]
    min_val = (-5, 0)
    max_val = (30, 15)
    layer = Surface((np.array(data), np.array((0, 1, 2)), np.array((0, 0, 0))))
    extent = np.array((min_val, max_val))
    check_layer_world_data_extent(layer, extent, (3, 1), (20, 5))


def test_shading():
    """Test setting shading"""
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    layer = Surface(data)

    # change shading property
    shading = 'flat'
    layer.shading = shading
    assert layer.shading == shading

    # set shading as keyword argument
    layer = Surface(data, shading=shading)
    assert layer.shading == shading
