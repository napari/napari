import numpy as np
import pytest

from napari.components.layerlist import LayerList
from napari.layers import Image, Points


@pytest.fixture
def layer_list():
    return LayerList()


@pytest.fixture
def points_layer():
    return Points()


@pytest.fixture
def image_layer():
    data = np.ones((10, 10))
    data[::2, ::2] = 0
    return Image(data)
