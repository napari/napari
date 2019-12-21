import numpy as np
from napari.layers import Labels


def test_random_pyramid():
    """Test instantiating Labels layer with random 2D pyramid data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data, is_pyramid=True)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_infer_pyramid():
    """Test instantiating Labels layer with random 2D pyramid data."""
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_3D_pyramid():
    """Test instantiating Labels layer with 3D data."""
    shapes = [(8, 40, 20), (4, 20, 10), (2, 10, 5)]
    np.random.seed(0)
    data = [np.random.randint(20, size=s) for s in shapes]
    layer = Labels(data, is_pyramid=True)
    assert layer.data == data
    assert layer.is_pyramid is True
    assert layer.editable is False
    assert layer.ndim == len(shapes[0])
    assert layer.shape == shapes[0]
    assert layer.rgb is False
    assert layer._data_view.ndim == 2


def test_create_random_pyramid():
    """Test instantiating Labels layer with random 2D data."""
    shape = (20_000, 20)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.is_pyramid is True
    assert layer.editable is False
    assert layer._data_pyramid[0].shape == shape
    assert layer._data_pyramid[1].shape == (shape[0] / 2, shape[1])
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.rgb is False
    assert layer._data_view.ndim == 2
