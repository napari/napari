"""Tests for components.chunk."""

import numpy as np

from ....layers.image import Image
from .. import ChunkKey


def test_chunk_key():

    data = np.random.random((512, 512))
    layer1 = Image(data)
    layer2 = Image(data)

    key1 = ChunkKey(layer1, (0, 0))
    key2 = ChunkKey(layer1, (0, 0))
    assert key1 == key2
    assert key1.key == key2.key

    assert key1.layer_id == id(layer1)
    assert key1.data_level == layer1.data_level

    key3 = ChunkKey(layer2, (0, 0))
    assert key1 != key3
    assert key2 != key3

    key4 = ChunkKey(layer2, (0, 1))
    assert key1 != key4
    assert key2 != key4
    assert key3 != key4

    key5 = ChunkKey(layer2, (0, 1))
    assert key1 != key5
    assert key2 != key5
    assert key3 != key5
    assert key4 == key5
