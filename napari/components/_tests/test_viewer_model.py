import numpy as np
import pytest

from napari.components import ViewerModel
from napari._tests.utils import good_layer_data


def test_viewer_model():
    """Test instantiating viewer model."""
    viewer = ViewerModel()
    assert viewer.title == 'napari'
    assert len(viewer.layers) == 0
    assert viewer.dims.ndim == 2

    # Create viewer model with custom title
    viewer = ViewerModel(title='testing')
    assert viewer.title == 'testing'


def test_add_image():
    """Test adding image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_volume():
    """Test adding volume."""
    viewer = ViewerModel(ndisplay=3)
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 3


def test_add_pyramid():
    """Test adding image pyramid."""
    viewer = ViewerModel()
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_image(data, is_pyramid=True)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_labels():
    """Test adding labels image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_points():
    """Test adding points."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_empty_points_to_empty_viewer():
    viewer = ViewerModel()
    pts = viewer.add_points(name='empty points')
    assert pts.dims.ndim == 2
    pts.add([1000.0, 27.0])
    assert pts.data.shape == (1, 2)


def test_add_empty_points_on_top_of_image():
    viewer = ViewerModel()
    image = np.random.random((8, 64, 64))
    # add_image always returns the corresponding layer
    _ = viewer.add_image(image)
    pts = viewer.add_points()
    assert pts.dims.ndim == 3
    pts.add([5.0, 32.0, 61.0])
    assert pts.data.shape == (1, 3)


def test_add_empty_shapes_layer():
    viewer = ViewerModel()
    image = np.random.random((8, 64, 64))
    # add_image always returns the corresponding layer
    _ = viewer.add_image(image)
    shp = viewer.add_shapes()
    assert shp.dims.ndim == 3


def test_add_vectors():
    """Test adding vectors."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_shapes():
    """Test adding shapes."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2


def test_add_surface():
    """Test adding 3D surface."""
    viewer = ViewerModel()
    np.random.seed(0)
    vertices = np.random.random((10, 3))
    faces = np.random.randint(10, size=(6, 3))
    values = np.random.random(10)
    data = (vertices, faces, values)
    viewer.add_surface(data)
    assert len(viewer.layers) == 1
    assert np.all(
        [np.all(vd == d) for vd, d in zip(viewer.layers[0].data, data)]
    )
    assert viewer.dims.ndim == 3


def test_mix_dims():
    """Test adding images of mixed dimensionality."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 2

    data = np.random.random((6, 10, 15))
    viewer.add_image(data)
    assert len(viewer.layers) == 2
    assert np.all(viewer.layers[1].data == data)
    assert viewer.dims.ndim == 3


def test_new_labels():
    """Test adding new labels layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer._new_labels()
    assert len(viewer.layers) == 1
    assert np.max(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2

    # Add labels with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert len(viewer.layers) == 2
    assert np.max(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2


def test_new_points():
    """Test adding new points layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer.add_points()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_points()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2


def test_new_shapes():
    """Test adding new shapes layer."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer.add_shapes()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0
    assert viewer.dims.ndim == 2

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer.add_shapes()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0
    assert viewer.dims.ndim == 2


def test_swappable_dims():
    """Test swapping dims after adding layers."""
    viewer = ViewerModel()
    np.random.seed(0)
    image_data = np.random.random((7, 12, 10, 15))
    viewer.add_image(image_data)
    assert np.all(viewer.layers['Image']._data_view == image_data[0, 0, :, :])

    points_data = np.random.randint(6, size=(10, 4))
    viewer.add_points(points_data)

    vectors_data = np.random.randint(6, size=(10, 2, 4))
    viewer.add_vectors(vectors_data)

    labels_data = np.random.randint(20, size=(7, 12, 10, 15))
    viewer.add_labels(labels_data)
    assert np.all(viewer.layers['Labels']._data_raw == labels_data[0, 0, :, :])

    # Swap dims
    viewer.dims.order = [0, 2, 1, 3]
    assert viewer.dims.order == [0, 2, 1, 3]
    assert np.all(viewer.layers['Image']._data_view == image_data[0, :, 0, :])
    assert np.all(viewer.layers['Labels']._data_raw == labels_data[0, :, 0, :])


def test_grid():
    "Test grid_view"
    viewer = ViewerModel()

    np.random.seed(0)
    # Add image
    for i in range(6):
        data = np.random.random((10, 15))
        viewer.add_image(data)
    assert np.all(viewer.grid_size == (1, 1))
    assert viewer.grid_stride == 1

    # enter grid view
    viewer.grid_view()
    assert np.all(viewer.grid_size == (3, 3))
    assert viewer.grid_stride == 1

    # retun to stack view
    viewer.stack_view()
    assert np.all(viewer.grid_size == (1, 1))
    assert viewer.grid_stride == 1

    # reenter grid view
    viewer.grid_view(n_column=2, n_row=3, stride=-2)
    assert np.all(viewer.grid_size == (3, 2))
    assert viewer.grid_stride == -2


def test_svg():
    "Test generating svg"
    viewer = ViewerModel()

    np.random.seed(0)
    # Add image
    data = np.random.random((10, 15))
    viewer.add_image(data)

    # Add labels
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)

    # Add points
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)

    # Add vectors
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)

    # Add shapes
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)

    # Generate svg
    svg = viewer.to_svg()
    assert type(svg) == str


def test_add_remove_layer_dims_change():
    """Test dims change appropriately when adding and removing layers."""
    np.random.seed(0)
    viewer = ViewerModel()

    # Check ndim starts at 2
    assert viewer.dims.ndim == 2

    # Check ndim increase to 3 when 3D data added
    data = np.random.random((10, 15, 20))
    layer = viewer.add_image(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)
    assert viewer.dims.ndim == 3

    # Remove layer and check ndim returns to 2
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0
    assert viewer.dims.ndim == 2


@pytest.mark.parametrize('data', good_layer_data)
def test_add_layer_from_data(data):
    # make sure adding valid layer data calls the proper corresponding add_*
    # method for all layer types
    viewer = ViewerModel()
    viewer._add_layer_from_data(*data)

    # make sure a layer of the correct type got added
    assert len(viewer.layers) == 1
    expected_layer_type = data[2] if len(data) > 2 else 'image'
    assert viewer.layers[0].__class__.__name__.lower() == expected_layer_type


def test_add_layer_from_data_raises():
    # make sure that adding invalid data or kwargs raises the right errors
    viewer = ViewerModel()
    # unrecognized layer type raises Value Error
    with pytest.raises(ValueError):
        # 'layer' is not a valid type
        # (even though there is an add_layer method)
        viewer._add_layer_from_data(
            np.random.random((10, 10)), layer_type='layer'
        )

    # even with the correct meta kwargs, the underlying add_* method may raise
    with pytest.raises(ValueError):
        # improper dims for rgb data
        viewer._add_layer_from_data(
            np.random.random((10, 10, 6)), {'rgb': True}
        )

    # using a kwarg in the meta dict that is invalid for the corresponding
    # add_* method raises a TypeError
    with pytest.raises(TypeError):
        viewer._add_layer_from_data(
            np.random.random((10, 2, 2)) * 20,
            {'rgb': True},  # vectors do not have an 'rgb' kwarg
            layer_type='vectors',
        )
