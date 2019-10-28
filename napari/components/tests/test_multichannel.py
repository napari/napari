import numpy as np
from napari.components import ViewerModel

base_colormaps = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']


def test_multichannel():
    """Test adding multichannel image."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer.add_image(data, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert np.all(viewer.layers[i].data == data.take(i, axis=-1))
        assert viewer.layers[i].colormap[0] == base_colormaps[i]


def test_specified_multichannel():
    """Test adding multichannel image with color channel set."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((5, 10, 15))
    viewer.add_image(data, channel_axis=0)
    assert len(viewer.layers) == data.shape[0]
    for i in range(data.shape[0]):
        assert np.all(viewer.layers[i].data == data.take(i, axis=0))


def test_names():
    """Test adding multichannel image with custom names."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    names = ['multi ' + str(i + 3) for i in range(data.shape[-1])]
    viewer.add_image(data, name=names, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].name == names[i]

    viewer = ViewerModel()
    name = 'example'
    names = [name] + [name + f' [{i + 1}]' for i in range(data.shape[-1] - 1)]
    viewer.add_image(data, name=name, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].name == names[i]


def test_colormaps():
    """Test adding multichannel image with custom colormaps."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    colormap = 'gray'
    viewer.add_image(data, colormap=colormap, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].colormap[0] == colormap

    viewer = ViewerModel()
    colormaps = ['gray', 'blue', 'red', 'green', 'yellow']
    viewer.add_image(data, colormap=colormaps, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].colormap[0] == colormaps[i]


def test_contrast_limits():
    """Test adding multichannel image with custom contrast limits."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    clims = [0.3, 0.7]
    viewer.add_image(data, contrast_limits=clims, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].contrast_limits == clims

    viewer = ViewerModel()
    clims = [[0.3, 0.7], [0.1, 0.9], [0.3, 0.9], [0.4, 0.9], [0.2, 0.9]]
    viewer.add_image(data, contrast_limits=clims, channel_axis=-1)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].contrast_limits == clims[i]


def test_gamma():
    """Test adding multichannel image with custom gamma."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    gamma = 0.7
    viewer.add_multichannel(data, gamma=gamma)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].gamma == gamma

    viewer = ViewerModel()
    gammas = [0.3, 0.4, 0.5, 0.6, 0.7]
    viewer.add_multichannel(data, gamma=gammas)
    assert len(viewer.layers) == data.shape[-1]
    for i in range(data.shape[-1]):
        assert viewer.layers[i].gamma == gammas[i]
