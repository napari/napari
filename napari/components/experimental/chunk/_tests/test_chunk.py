"""Tests for components.experimental.chunk."""
import numpy as np
import pytest

from napari.components.experimental.chunk import (
    ChunkKey,
    LayerKey,
    LayerRef,
    chunk_loader,
)
from napari.layers.image import Image
from napari.utils import config


def _create_layer() -> Image:
    """Return a small random Image layer."""
    data = np.random.random((32, 16))
    return Image(data)


def test_chunk_key():
    """Test the ChunkKey class."""

    layer1 = _create_layer()
    layer2 = _create_layer()

    layer_key1 = LayerKey.from_layer(layer1, (0, 0))
    layer_key2 = LayerKey.from_layer(layer2, (0, 0))

    # key1 and key2 should be identical.
    key1 = ChunkKey(layer_key1)
    key1b = ChunkKey(layer_key1)
    assert key1 == key1b
    assert key1.key == key1b.key

    # Check key1 attributes.
    assert key1.layer_key.layer_id == id(layer1)
    assert key1.layer_key.data_level == layer1.data_level

    # key2 is for a different layer.
    key2 = ChunkKey(layer_key2)
    assert key1 != key2
    assert key1b != key2

    # key4 has different indices.
    layer_key3 = LayerKey.from_layer(layer2, (0, 1))
    key3 = ChunkKey(layer_key3)
    assert key1 != key3
    assert key1b != key3
    assert key2 != key3

    # key4 matches key3.
    key4 = ChunkKey(layer_key3)
    assert key1 != key4
    assert key1b != key4
    assert key2 != key4
    assert key3 == key4


def test_loader():
    """Test ChunkRequest and the ChunkLoader."""
    if not config.async_loading:
        return  # temporary until we add the @async_only pytest mark

    layer = _create_layer()
    layer_key = LayerKey.from_layer(layer, (0, 0))
    key = ChunkKey(layer_key)

    shape = (64, 32)
    transpose_shape = (32, 64)

    # Just load one array.
    data = np.random.random(shape)
    chunks = {'image': data}

    # Give data2 different data.
    data2 = data * 2

    # Create the ChunkRequest.
    layer_ref = LayerRef.create_from_layer(layer, (0, 0))
    request = chunk_loader.create_request(layer_ref, key, chunks)

    # Should be compatible with the layer we made it from!
    # assert request.is_compatible(layer)

    # Load the ChunkRequest.
    request = chunk_loader.load_chunk(request)

    # Data should only match data not data2.
    assert np.all(data == request.image.data)
    assert not np.all(data2 == request.image.data)

    # request.image is just short-hand for request.chunks['image']
    assert np.all(request.image.data == request.chunks['image'].data)

    # Since we didn't ask for a thumbnail_source it should be the image.
    assert np.all(request.thumbnail_source.data == request.image.data)

    # KeyError for chunks that do not exist.
    with pytest.raises(KeyError):
        request.chunks['missing_chunk_name']

    # Test transpose_chunks()
    request.transpose_chunks((1, 0))
    assert request.image.shape == transpose_shape
