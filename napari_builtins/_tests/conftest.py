from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from npe2 import DynamicPlugin, PluginManager, PluginManifest

import napari_builtins
from napari import layers

try:
    from skimage.data import image_fetcher
except ImportError:
    from skimage.data import data_dir

    class image_fetcher:  # type: ignore
        @staticmethod
        def fetch(data_name):
            if data_name.startswith("data/"):
                data_name = data_name[5:]
            if (path := Path(data_dir) / data_name).exists():
                return path
            raise ValueError(
                f"Legacy skimage image_fetcher cannot find file: {path}"
            )


@pytest.fixture(autouse=True)
def _mock_npe2_pm():
    """Mock plugin manager with no registered plugins."""
    with patch.object(PluginManager, 'discover'):
        _pm = PluginManager()
    with patch('npe2.PluginManager.instance', return_value=_pm):
        yield _pm


@pytest.fixture(autouse=True)
def _use_builtins(_mock_npe2_pm: PluginManager):

    plugin = DynamicPlugin('napari', plugin_manager=_mock_npe2_pm)
    mf = PluginManifest.from_file(
        Path(napari_builtins.__file__).parent / 'builtins.yaml'
    )
    plugin.manifest = mf
    with plugin:
        yield plugin


LAYERS = [
    layers.Image(np.random.rand(10, 10)),
    layers.Labels(np.random.randint(0, 16000, (32, 32), 'uint64')),
    layers.Points(np.random.rand(20, 2)),
    layers.Points(
        np.random.rand(20, 2), properties={'values': np.random.rand(20)}
    ),
    layers.Shapes(
        [
            np.random.rand(2, 2),
            np.random.rand(2, 2),
            np.random.rand(6, 2),
            np.random.rand(6, 2),
            np.random.rand(2, 2),
        ],
        shape_type=['ellipse', 'line', 'path', 'polygon', 'rectangle'],
    ),
]


@pytest.fixture(params=LAYERS)
def some_layer(request):
    return request.param


@pytest.fixture
def two_pngs():
    return [image_fetcher.fetch(f'data/{n}.png') for n in ('moon', 'camera')]


@pytest.fixture
def rgb_png():
    return [image_fetcher.fetch('data/astronaut.png')]


@pytest.fixture
def single_png():
    return [image_fetcher.fetch('data/camera.png')]


@pytest.fixture
def irregular_images():
    return [image_fetcher.fetch(f'data/{n}.png') for n in ('camera', 'coins')]


@pytest.fixture
def single_tiff():
    return [image_fetcher.fetch('data/multipage.tif')]
