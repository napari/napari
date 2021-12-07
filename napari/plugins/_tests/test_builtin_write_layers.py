import os

import numpy as np

from napari._tests.utils import assert_layer_state_equal


# the layer_writer_and_data fixture is defined in napari/conftest.py
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
    read_data, read_meta, layer_type = reader(path)

    # Compare read data to original data on layer
    if type(read_data) is list:
        for rd, ld in zip(read_data, layer_data[0]):
            np.testing.assert_allclose(rd, ld)
    else:
        np.testing.assert_allclose(read_data, layer_data[0])

    # Instantiate layer
    read_layer = Layer(read_data, **read_meta)
    read_layer_data = read_layer.as_layer_data_tuple()

    # Compare layer data
    if type(read_layer_data[0]) is list:
        for ld, rld in zip(layer_data[0], read_layer_data[0]):
            np.testing.assert_allclose(ld, rld)
    else:
        np.testing.assert_allclose(layer_data[0], read_layer_data[0])

    # Compare layer metadata
    assert_layer_state_equal(layer_data[1], read_layer_data[1])

    # Compare layer type
    assert layer_data[2] == read_layer_data[2]


# the layer_writer_and_data fixture is defined in napari/conftest.py
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


# the layer_writer_and_data fixture is defined in napari/conftest.py
def test_no_write_layer_bad_extension(tmpdir, layer_writer_and_data):
    """Test not writing layer data with a bad extension."""
    writer, layer_data, _, _, _ = layer_writer_and_data
    path = os.path.join(tmpdir, 'layer_file.bad_extension')

    # Check file does not exist
    assert not os.path.isfile(path)

    # Check no data is written
    assert not writer(path, layer_data[0], layer_data[1])

    # Check file still does not exist
    assert not os.path.isfile(path)


# the layer_writer_and_data fixture is defined in napari/conftest.py
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
