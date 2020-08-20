"""Tests for components.chunk."""

import numpy as np
import pytest

from ....layers.image import Image
from .. import ChunkKey, chunk_loader


def _create_layer() -> Image:
    """Return a small random Image layer."""
    data = np.random.random((16, 16))
    return Image(data)


def test_chunk_key():
    """Test the ChunkKey class."""

    layer1 = _create_layer()
    layer2 = _create_layer()

    # key1 and key2 should be identical.
    key1 = ChunkKey(layer1, (0, 0))
    key2 = ChunkKey(layer1, (0, 0))
    assert key1 == key2
    assert key1.key == key2.key

    # Check key1 attributes.
    assert key1.layer_id == id(layer1)
    assert key1.data_level == layer1.data_level

    # key3 is for a different layer.
    key3 = ChunkKey(layer2, (0, 0))
    assert key1 != key3
    assert key2 != key3

    # key4 has different indices.
    key4 = ChunkKey(layer2, (0, 1))
    assert key1 != key4
    assert key2 != key4
    assert key3 != key4

    # key5 matches key4.
    key5 = ChunkKey(layer2, (0, 1))
    assert key1 != key5
    assert key2 != key5
    assert key3 != key5
    assert key4 == key5


def test_loader():
    """Test ChunkRequest and the ChunkLoader."""
    layer = _create_layer()
    key = ChunkKey(layer, (0, 0))

    # Just load one array.
    data = np.random.random((64, 64))
    chunks = {'image': data}

    # Give data2 different data.
    data2 = data * 2

    # Create the ChunkRequest.
    request = chunk_loader.create_request(layer, key, chunks)

    # Load the ChunkRequest.
    request = chunk_loader.load_chunk(request)

    # Data should only match data not data2.
    assert np.all(data[:] == request.image.data[:])
    assert not np.all(data2[:] == request.image.data[:])

    # request.image is just short-hand for request.chunks['image']
    assert np.all(request.image.data == request.chunks['image'].data)

    # Since we didn't ask for a thumbnail_source it should be the image.
    assert np.all(request.thumbnail_source.data[:] == request.image.data[:])

    # KeyError for chunks that do not exist.
    with pytest.raises(KeyError):
        request.chunks['missing_chunk_name']
