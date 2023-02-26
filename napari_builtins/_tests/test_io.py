import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple, Tuple
from uuid import uuid4

import dask.array as da
import imageio
import npe2
import numpy as np
import pytest
import tifffile
import zarr

from napari_builtins.io._read import (
    _guess_layer_type_from_column_names,
    _guess_zarr_path,
    csv_to_layer_data,
    magic_imread,
    read_csv,
)
from napari_builtins.io._write import write_csv


class ImageSpec(NamedTuple):
    shape: Tuple[int, ...]
    dtype: str
    ext: str
    levels: int = 1


PNG = ImageSpec((10, 10), 'uint8', '.png')
PNG_RGB = ImageSpec((10, 10, 3), 'uint8', '.png')
PNG_RECT = ImageSpec((10, 15), 'uint8', '.png')
TIFF_2D = ImageSpec((15, 10), 'uint8', '.tif')
TIFF_3D = ImageSpec((2, 15, 10), 'uint8', '.tif')
ZARR1 = ImageSpec((10, 20, 20), 'uint8', '.zarr')


@pytest.fixture
def _write_spec(tmp_path: Path):
    def writer(spec: ImageSpec):
        image = np.random.random(spec.shape).astype(spec.dtype)
        fname = tmp_path / f'{uuid4()}{spec.ext}'
        if spec.ext == '.tif':
            tifffile.imwrite(str(fname), image)
        elif spec.ext == '.zarr':
            fname.mkdir()
            z = zarr.open(str(fname), 'a', shape=image.shape)
            z[:] = image
        else:
            imageio.imwrite(str(fname), image)
        return fname

    return writer


def test_no_files_raises(tmp_path):
    with pytest.raises(ValueError) as e:
        magic_imread(tmp_path)
    assert "No files found in" in str(e.value)


def test_guess_zarr_path():
    assert _guess_zarr_path('dataset.zarr')
    assert _guess_zarr_path('dataset.zarr/some/long/path')
    assert not _guess_zarr_path('data.tif')
    assert not _guess_zarr_path('no_zarr_suffix/data.png')


def test_zarr():
    image = np.random.random((10, 20, 20))
    with TemporaryDirectory(suffix='.zarr') as fout:
        z = zarr.open(fout, 'a', shape=image.shape)
        z[:] = image
        image_in = magic_imread([fout])
        # Note: due to lazy loading, the next line needs to happen within
        # the context manager. Alternatively, we could convert to NumPy here.
        np.testing.assert_array_equal(image, image_in)


def test_zarr_nested(tmp_path):
    image = np.random.random((10, 20, 20))
    image_name = 'my_image'
    root_path = tmp_path / 'dataset.zarr'
    grp = zarr.open(str(root_path), mode='a')
    grp.create_dataset(image_name, data=image)

    image_in = magic_imread([str(root_path / image_name)])
    np.testing.assert_array_equal(image, image_in)


def test_zarr_multiscale():
    multiscale = [
        np.random.random((20, 20)),
        np.random.random((10, 10)),
        np.random.random((5, 5)),
    ]
    with TemporaryDirectory(suffix='.zarr') as fout:
        root = zarr.open_group(fout, 'a')
        for i in range(len(multiscale)):
            shape = 20 // 2**i
            z = root.create_dataset(str(i), shape=(shape,) * 2)
            z[:] = multiscale[i]
        multiscale_in = magic_imread([fout])
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
    write_csv(expected_filename, expected_data, column_names=column_names)
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
    write_csv(expected_filename, expected_data, column_names=column_names)
    assert os.path.exists(expected_filename)

    # Read csv file
    read_data, read_column_names, _ = read_csv(expected_filename)
    read_data = np.array(read_data).astype('float')
    np.testing.assert_allclose(expected_data, read_data)

    assert column_names == read_column_names


def test_guess_layer_type_from_column_names():
    points_names = ['index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(points_names) == 'points'

    shapes_names = ['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(shapes_names) == 'shapes'

    also_points_names = ['no-index', 'axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(also_points_names) == 'points'

    bad_names = ['no-index', 'no-axis-0', 'axis-1']
    assert _guess_layer_type_from_column_names(bad_names) is None


def test_read_csv_raises(tmp_path):
    """Test various exception raising circumstances with read_csv."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type = None, any, points
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert read_csv(temp, require_type=None)[2] == 'points'
    assert read_csv(temp, require_type='any')[2] == 'points'
    assert read_csv(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError):
        read_csv(temp, require_type='shapes')

    # test that unrecognized data is detected with require_type = None
    # but raises for specific shape types or "any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert read_csv(temp, require_type=None)[2] is None
    with pytest.raises(ValueError):
        assert read_csv(temp, require_type='any')
    with pytest.raises(ValueError):
        assert read_csv(temp, require_type='points')
    with pytest.raises(ValueError):
        read_csv(temp, require_type='shapes')


def test_csv_to_layer_data_raises(tmp_path):
    """Test various exception raising circumstances with csv_to_layer_data."""
    temp = tmp_path / 'points.csv'

    # test that points data is detected with require_type == points, any, None
    # but raises for other shape types.
    data = [['index', 'axis-0', 'axis-1']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert csv_to_layer_data(temp, require_type=None)[2] == 'points'
    assert csv_to_layer_data(temp, require_type='any')[2] == 'points'
    assert csv_to_layer_data(temp, require_type='points')[2] == 'points'
    with pytest.raises(ValueError):
        csv_to_layer_data(temp, require_type='shapes')

    # test that unrecognized data simply returns None when require_type==None
    # but raises for specific shape types or require_type=="any"
    data = [['some', 'random', 'header']]
    data.extend(np.random.random((3, 3)).tolist())
    with open(temp, mode='w', newline='') as csvfile:
        csv.writer(csvfile).writerows(data)
    assert csv_to_layer_data(temp, require_type=None) is None
    with pytest.raises(ValueError):
        assert csv_to_layer_data(temp, require_type='any')
    with pytest.raises(ValueError):
        assert csv_to_layer_data(temp, require_type='points')
    with pytest.raises(ValueError):
        csv_to_layer_data(temp, require_type='shapes')


@pytest.mark.parametrize('spec', [PNG, PNG_RGB, TIFF_3D, TIFF_2D])
@pytest.mark.parametrize('stacks', [1, 3])
def test_single_file(spec: ImageSpec, _write_spec, stacks: int):
    fnames = [str(_write_spec(spec)) for _ in range(stacks)]
    [(layer_data,)] = npe2.read(fnames, stack=stacks > 1)
    assert isinstance(layer_data, np.ndarray if stacks == 1 else da.Array)
    assert layer_data.shape == tuple(
        i for i in (stacks,) + spec.shape if i > 1
    )
    assert layer_data.dtype == spec.dtype


@pytest.mark.parametrize(
    'spec', [PNG, [PNG], [PNG, PNG], TIFF_3D, [TIFF_3D, TIFF_3D]]
)
@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('use_dask', [True, False, None])
def test_magic_imread(_write_spec, spec: ImageSpec, stack, use_dask):
    fnames = (
        [_write_spec(s) for s in spec]
        if isinstance(spec, list)
        else _write_spec(spec)
    )
    images = magic_imread(fnames, stack=stack, use_dask=use_dask)
    if isinstance(spec, ImageSpec):
        expect_shape = spec.shape
    else:
        expect_shape = (len(spec),) + spec[0].shape if stack else spec[0].shape
    expect_shape = tuple(i for i in expect_shape if i > 1)

    expected_arr_type = (
        da.Array
        if (
            use_dask
            or (use_dask is None and isinstance(spec, list) and len(spec) > 1)
        )
        else np.ndarray
    )
    if isinstance(spec, list) and len(spec) > 1 and not stack:
        assert isinstance(images, list)
        assert all(isinstance(img, expected_arr_type) for img in images)
        assert all(img.shape == expect_shape for img in images)
    else:
        assert isinstance(images, expected_arr_type)
        assert images.shape == expect_shape


@pytest.mark.parametrize('stack', [True, False])
def test_irregular_images(_write_spec, stack):
    specs = [PNG, PNG_RECT]
    fnames = [str(_write_spec(spec)) for spec in specs]

    # Ideally, this would work "magically" with dask and irregular images,
    # but there is no foolproof way to do this without reading in all the
    # files. We need to be able to inspect the file shape without reading
    # it in first, then we can automatically turn stacking off when shapes
    # are irregular (and create proper dask arrays)
    if stack:
        with pytest.raises(
            ValueError, match='input arrays must have the same shape'
        ):
            magic_imread(fnames, use_dask=False, stack=stack)
        return

    images = magic_imread(fnames, use_dask=False, stack=stack)
    assert isinstance(images, list)
    assert len(images) == 2
    assert all(img.shape == spec.shape for img, spec in zip(images, specs))


def test_add_zarr(_write_spec):
    [out] = npe2.read([str(_write_spec(ZARR1))], stack=False)
    assert out[0].shape == ZARR1.shape  # type: ignore


def test_add_zarr_1d_array_is_ignored():
    # For more details: https://github.com/napari/napari/issues/1471
    with TemporaryDirectory(suffix='.zarr') as zarr_dir:
        z = zarr.open(zarr_dir, 'w')
        z['1d'] = np.zeros(3)

        image_path = os.path.join(zarr_dir, '1d')
        assert npe2.read([image_path], stack=False) == [(None,)]


def test_add_many_zarr_1d_array_is_ignored():
    # For more details: https://github.com/napari/napari/issues/1471
    with TemporaryDirectory(suffix='.zarr') as zarr_dir:
        z = zarr.open(zarr_dir, 'w')
        z['1d'] = np.zeros(3)
        z['2d'] = np.zeros((3, 4))
        z['3d'] = np.zeros((3, 4, 5))

        for name in z.array_keys():
            [out] = npe2.read([os.path.join(zarr_dir, name)], stack=False)
            if name == '1d':
                assert out == (None,)
            else:
                assert isinstance(out[0], da.Array)
                assert out[0].ndim == int(name[0])
