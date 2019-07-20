import numpy as np
from napari.components import ViewerModel


def test_viewer_model():
    """Test instantiating viewer model."""
    viewer = ViewerModel()
    assert viewer.title == 'napari'
    assert len(viewer.layers) == 0

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


def test_add_volume():
    """Test adding volume."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15, 20))
    viewer.add_volume(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_add_pyramid():
    """Test adding image pyramid."""
    viewer = ViewerModel()
    shapes = [(40, 20), (20, 10), (10, 5)]
    np.random.seed(0)
    data = [np.random.random(s) for s in shapes]
    viewer.add_pyramid(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_add_labels():
    """Test adding labels image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    viewer.add_labels(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_add_points():
    """Test adding points."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    viewer.add_points(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_add_vectors():
    """Test adding vectors."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 2, 2))
    viewer.add_vectors(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_add_shapes():
    """Test adding vectors."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    viewer.add_shapes(data)
    assert len(viewer.layers) == 1
    assert np.all(viewer.layers[0].data == data)


def test_new_labels():
    """Test adding new labels image."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer._new_labels()
    assert len(viewer.layers) == 1
    assert np.max(viewer.layers[0].data) == 0

    # Add labels with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_labels()
    assert len(viewer.layers) == 2
    assert np.max(viewer.layers[1].data) == 0


def test_new_points():
    """Test adding new points image."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer._new_points()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_points()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0


def test_new_shapes():
    """Test adding new shapes image."""
    # Add labels to empty viewer
    viewer = ViewerModel()
    viewer._new_shapes()
    assert len(viewer.layers) == 1
    assert len(viewer.layers[0].data) == 0

    # Add points with image already present
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 15))
    viewer.add_image(data)
    viewer._new_shapes()
    assert len(viewer.layers) == 2
    assert len(viewer.layers[1].data) == 0
