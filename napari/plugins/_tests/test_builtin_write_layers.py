import os
import numpy as np
import pytest

from napari.layers import Image, Points
from napari.plugins._builtins import napari_write_image, napari_write_points
from napari.utils import io


@pytest.fixture(params=['image', 'points', 'points-with-properties'])
def layer_writer_and_data(request):
    if request.param == 'image':
        data = np.random.rand(20, 20)
        Layer = Image
        layer = Image(data)
        writer = napari_write_image
        extension = '.tif'

        def reader(path):
            return (
                io.imread(path),
                {},
            )

    elif request.param == 'points':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data)
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {},
            )

    elif request.param == 'points-with-properties':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data, properties={'values': np.random.rand(20)})
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {
                    'properties': {
                        io.read_csv(path)[1][3]: io.read_csv(path)[0][:, 3]
                    }
                },
            )

    else:
        return None, None, None, None, None

    layer_data = layer.as_layer_data_tuple()
    return writer, layer_data, extension, reader, Layer


def test_write_layer_with_round_trip(tmpdir, layer_writer_and_data):
    """Test writing layer data from napari layer_data tuple."""
    writer, layer_data, extension, reader, Layer = layer_writer_and_data
    path = os.path.join(tmpdir, 'layer_file' + extension)

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    assert writer(path, layer_data[0], layer_data[1])

    # Check file now exists
    assert os.path.isfile(path)

    # Read data
    read_data, read_meta = reader(path)

    # Compare read data to original data on layer
    np.testing.assert_allclose(read_data, layer_data[0])

    # Instantiate layer
    read_layer = Layer(read_data, **read_meta)
    read_layer_data = read_layer.as_layer_data_tuple()

    # Compare layer data
    np.testing.assert_allclose(layer_data[0], read_layer_data[0])
    # # Compare layer metadata
    np.testing.assert_equal(layer_data[1], read_layer_data[1])
    # Compare layer type
    assert layer_data[2] == read_layer_data[2]


def test_write_layer_no_extension(tmpdir, layer_writer_and_data):
    """Test writing layer data with no extension."""
    writer, layer_data, extension, _, _ = layer_writer_and_data
    path = os.path.join(tmpdir, 'layer_file')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    assert writer(path, layer_data[0], layer_data[1])

    # Check file now exists with extension
    assert os.path.isfile(path + extension)


def test_no_write_layer_bad_extension(tmpdir, layer_writer_and_data):
    """Test not writing layer data with a bad extension."""
    writer, layer_data, _, _, _ = layer_writer_and_data
    path = os.path.join(tmpdir, 'layer_file.bad_extension')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Check no data is writen
    assert not writer(path, layer_data[0], layer_data[1])

    # Check file still does not exist
    assert not os.path.isfile(path)


def test_write_layer_no_metadata(tmpdir, layer_writer_and_data):
    """Test writing layer data with no metadata."""
    writer, layer_data, extension, _, _ = layer_writer_and_data
    path = os.path.join(tmpdir, 'layer_file' + extension)

    # Check file does not exist
    assert not os.path.isfile(path)

    # Write data
    assert writer(path, layer_data[0], {})

    # Check file now exists
    assert os.path.isfile(path)
