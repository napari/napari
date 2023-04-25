import numpy as np

from napari.layers import Labels


def test_random_multiscale():
    """Test instantiating Labels layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shapes[0]]
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_multiscale():
    """Test instantiating Labels layer with random 2D multiscale data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shapes[0]]
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_3D_multiscale():
    """Test instantiating Labels layer with 3D data."""
    shapes = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data, multiscale=True)
    assert layer.data == data
    assert layer.multiscale is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    np.testing.assert_array_equal(
        layer.extent.data[1], [s - 1 for s in shapes[0]]
    )
    assert layer.rgb is False
    assert layer._data_view.ndim == 2
