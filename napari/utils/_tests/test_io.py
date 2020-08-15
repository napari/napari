import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from dask import array as da

from napari.utils import io

try:
    import zarr

    zarr_available = True
except ImportError:
    zarr_available = False


# the following fixtures are defined in napari/conftest.py
# single_png, two_pngs, irregular_images, single_tiff


def test_single_png_defaults(single_png):
    image_files = single_png
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_single_png_single_file(single_png):
    image_files = single_png[0]
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_single_png_pathlib(single_png):
    image_files = Path(single_png[0])
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (512, 512)


def test_multi_png_defaults(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (2, 512, 512)


def test_multi_png_pathlib(two_pngs):
    image_files = [Path(png) for png in two_pngs]
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (2, 512, 512)


def test_multi_png_no_dask(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files, use_dask=False)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 512, 512)


def test_multi_png_no_stack(two_pngs):
    image_files = two_pngs
    images = io.magic_imread(image_files, stack=False)
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(a.shape == (512, 512) for a in images)


def test_no_files_raises(tmp_path, two_pngs):
    with pytest.raises(ValueError) as e:
        io.magic_imread(tmp_path)
    assert "No files found in" in str(e.value)


def test_irregular_images(irregular_images):
    image_files = irregular_images
    # Ideally, this would work "magically" with dask and irregular images,
    # but there is no foolproof way to do this without reading in all the
    # files. We need to be able to inspect the file shape without reading
    # it in first, then we can automatically turn stacking off when shapes
    # are irregular (and create proper dask arrays)
    images = io.magic_imread(image_files, use_dask=False, stack=False)
    assert isinstance(images, list)
    assert len(images) == 2
    assert tuple(image.shape for image in images) == ((512, 512), (303, 384))


def test_tiff(single_tiff):
    image_files = single_tiff
    images = io.magic_imread(image_files)
    assert isinstance(images, np.ndarray)
    assert images.shape == (2, 15, 10)
    assert images.dtype == np.uint8


def test_many_tiffs(single_tiff):
    image_files = single_tiff * 3
    images = io.magic_imread(image_files)
    assert isinstance(images, da.Array)
    assert images.shape == (3, 2, 15, 10)
    assert images.dtype == np.uint8


def test_single_filename(single_tiff):
    image_files = single_tiff[0]
    images = io.magic_imread(image_files)
    assert images.shape == (2, 15, 10)


def test_guess_zarr_path():
    assert io.guess_zarr_path('dataset.zarr')
    assert io.guess_zarr_path('dataset.zarr/some/long/path')
    assert not io.guess_zarr_path('data.tif')
    assert not io.guess_zarr_path('no_zarr_suffix/data.png')


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr():
    image = np.random.random((10, 20, 20))
    with TemporaryDirectory(suffix='.zarr') as fout:
        z = zarr.open(fout, 'a', shape=image.shape)
        z[:] = image
        image_in = io.magic_imread([fout])
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        np.testing.assert_array_equal(image, image_in)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr_nested(tmp_path):
    image = np.random.random((10, 20, 20))
    image_name = 'my_image'
    root_path = tmp_path / 'dataset.zarr'
    grp = zarr.open(str(root_path), mode='a')
    grp.create_dataset(image_name, data=image)

    image_in = io.magic_imread([str(root_path / image_name)])
    np.testing.assert_array_equal(image, image_in)


@pytest.mark.skipif(not zarr_available, reason='zarr not installed')
def test_zarr_multiscale():
    multiscale = [
        np.random.random((20, 20)),
        np.random.random((10, 10)),
        np.random.random((5, 5)),
    ]
    with TemporaryDirectory(suffix='.zarr') as fout:
        root = zarr.open_group(fout, 'a')
        for i in range(len(multiscale)):
            shape = 20 // 2 ** i
            z = root.create_dataset(str(i), shape=(shape,) * 2)
            z[:] = multiscale[i]
        multiscale_in = io.magic_imread([fout])
        assert len(multiscale) == len(multiscale_in)
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        for images, images_in in zip(multiscale, multiscale_in):
            np.testing.assert_array_equal(images, images_in)


def test_write_csv(tmpdir):
    expected_filename = os.path.join(tmpdir, 'test.csv')
    column_names = ['column_1', 'column_2', 'column_3']
    expected_data = np.random.random((5, len(column_names)))

    # Write csv file
    io.write_csv(expected_filename, expected_data, column_names=column_names)
    assert os.path.exists(expected_filename)

    # Check csv file is as expected
    with open(expected_filename) as output_csv:
        csv.reader(output_csv, delimiter=',')
        for row_index, row in enumerate(output_csv):
            if row_index == 0:
                assert row == "column_1,column_2,column_3\n"
            else:
                output_row_data = [float(i) for i in row.split(',')]
                np.testing.assert_allclose(
                    np.array(output_row_data), expected_data[row_index - 1]
                )


def test_read_csv(tmpdir):
    expected_filename = os.path.join(tmpdir, 'test.csv')
    column_names = ['column_1', 'column_2', 'column_3']
    expected_data = np.random.random((5, len(column_names)))

    # Write csv file
    io.write_csv(expected_filename, expected_data, column_names=column_names)
    assert os.path.exists(expected_filename)

    # Read csv file
    read_data, read_column_names, _ = io.read_csv(expected_filename)
    read_data = np.array(read_data).astype('float')
    np.testing.assert_allclose(expected_data, read_data)

    assert column_names == read_column_names


def test_guess_layer_type_from_column_names():
    points_names = ['index', 'axis-0', 'axis-1']
    assert io.guess_layer_type_from_column_names(points_names) == 'points'

    shapes_names = ['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1']
    assert io.guess_layer_type_from_column_names(shapes_names) == 'shapes'

    also_points_names = ['no-index', 'axis-0', 'axis-1']
    assert io.guess_layer_type_from_column_names(also_points_names) == 'points'

    bad_names = ['no-index', 'no-axis-0', 'axis-1']
    assert io.guess_layer_type_from_column_names(bad_names) is None


def test_read_csv_raises(tmp_path):
    """Test various exception raising circumstances with read_csv."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type = None, any, points
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert io.read_csv(temp, require_type=None)[2] == 'points'
    assert io.read_csv(temp, require_type='any')[2] == 'points'
    assert io.read_csv(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError):
        io.read_csv(temp, require_type='shapes')

    # test that unrecognized data is detected with require_type = None
    # but raises for specific shape types or "any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert io.read_csv(temp, require_type=None)[2] is None
    with pytest.raises(ValueError):
        assert io.read_csv(temp, require_type='any')
    with pytest.raises(ValueError):
        assert io.read_csv(temp, require_type='points')
    with pytest.raises(ValueError):
        io.read_csv(temp, require_type='shapes')


def test_csv_to_layer_data_raises(tmp_path):
    """Test various exception raising circumstances with csv_to_layer_data."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type == points, any, None
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert io.csv_to_layer_data(temp, require_type=None)[2] == 'points'
    assert io.csv_to_layer_data(temp, require_type='any')[2] == 'points'
    assert io.csv_to_layer_data(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError):
        io.csv_to_layer_data(temp, require_type='shapes')

    # test that unrecognized data simply returns None when require_type==None
    # but raises for specific shape types or require_type=="any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert io.csv_to_layer_data(temp, require_type=None) is None
    with pytest.raises(ValueError):
        assert io.csv_to_layer_data(temp, require_type='any')
    with pytest.raises(ValueError):
        assert io.csv_to_layer_data(temp, require_type='points')
    with pytest.raises(ValueError):
        io.csv_to_layer_data(temp, require_type='shapes')
