import numpy as np

from napari.components import ViewerModel


def test_translated_images():
    """Test two translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data, translate=[10, 0, 0])
    assert viewer.dims.range[0] == (0, 20, 1)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == [20, 10, 10]
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.current_step[0] == i


def test_scaled_images():
    """Test two scaled images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data[::2], scale=[2, 1, 1])
    assert viewer.dims.range[0] == (0, 10, 1)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == [10, 10, 10]
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.current_step[0] == i


def test_scaled_and_translated_images():
    """Test scaled and translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data[::2], scale=[2, 1, 1], translate=[10, 0, 0])
    assert viewer.dims.range[0] == (0, 20, 1)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == [20, 10, 10]
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.current_step[0] == i


def test_both_scaled_and_translated_images():
    """Test both scaled and translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data, scale=[2, 1, 1])
    viewer.add_image(data, scale=[2, 1, 1], translate=[20, 0, 0])
    assert viewer.dims.range[0] == (0, 40, 2)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == [20, 10, 10]
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.current_step[0] == i
