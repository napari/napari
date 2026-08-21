import itertools

import numpy as np
import pytest

pytest.importorskip('zarr.experimental', reason='requires zarr v3')

from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset,
    mandelbulb_dataset,
    mandelbulb_rgb_dataset,
    open_ome_zarr,
)


def test_mandelbrot_dataset():
    dataset = mandelbrot_dataset(max_levels=4, tilesize=32)
    arrays = dataset['arrays']
    assert len(arrays) == 4
    assert arrays[0].shape == (32 * 2**4,) * 2
    for fine, coarse in itertools.pairwise(arrays):
        assert fine.shape[0] == coarse.shape[0] * 2
    # data is readable and nontrivial
    coarsest = arrays[-1][:]
    assert coarsest.max() > 0


def test_mandelbrot_dataset_cache_consistency():
    dataset = mandelbrot_dataset(max_levels=3, tilesize=16)
    arr = dataset['arrays'][1]
    np.testing.assert_array_equal(arr[:], arr[:])


def test_mandelbulb_dataset():
    dataset = mandelbulb_dataset(max_levels=3, tilesize=8, maxiter=32)
    arrays = dataset['arrays']
    assert len(arrays) == 3
    assert arrays[0].ndim == 3
    assert arrays[-1][:].max() > 0


def test_mandelbulb_rgb_dataset():
    dataset = mandelbulb_rgb_dataset(max_levels=3, tilesize=8, maxiter=32)
    arrays = dataset['arrays']
    assert len(arrays) == 3
    # trailing length-3 channel axis (RGB)
    assert arrays[0].ndim == 4
    assert arrays[0].shape[-1] == 3
    assert arrays[0].dtype == np.uint8
    chunk = np.asarray(arrays[-1][:])
    assert chunk.shape[-1] == 3
    assert chunk.max() > 0


def _write_multiscale_ome_zarr(root_path, *, extra_translate=False):
    """Write a tiny 2-level OME-Zarr group with multiscales metadata."""
    import zarr

    group = zarr.open_group(str(root_path), mode='w', zarr_format=2)
    for level, size in enumerate((16, 8)):
        arr = group.create_array(
            name=str(level), shape=(size, size), chunks=(4, 4), dtype='u1'
        )
        arr[:] = level + 1
    transforms0 = [{'type': 'scale', 'scale': [2.0, 2.0]}]
    if extra_translate:
        transforms0.append({'type': 'translation', 'translation': [5.0, 7.0]})
    group.attrs['multiscales'] = [
        {
            'datasets': [
                {'path': '0', 'coordinateTransformations': transforms0},
                {
                    'path': '1',
                    'coordinateTransformations': [
                        {'type': 'scale', 'scale': [4.0, 4.0]}
                    ],
                },
            ],
        }
    ]
    return group


def test_open_ome_zarr_local_multiscale(tmp_path):
    root = tmp_path / 'sample.zarr'
    _write_multiscale_ome_zarr(root, extra_translate=True)

    arrays, scale, translate = open_ome_zarr(str(root))

    assert len(arrays) == 2
    assert arrays[0].shape == (16, 16)
    assert arrays[1].shape == (8, 8)
    assert scale == [2.0, 2.0]
    assert translate == [5.0, 7.0]


def test_open_ome_zarr_num_levels_and_no_translate(tmp_path):
    root = tmp_path / 'sample.zarr'
    _write_multiscale_ome_zarr(root, extra_translate=False)

    arrays, scale, translate = open_ome_zarr(str(root), num_levels=1)

    assert len(arrays) == 1
    assert scale == [2.0, 2.0]
    assert translate is None


def test_open_ome_zarr_nested_group(tmp_path):
    """multiscales metadata one group down is still discovered."""
    import zarr

    root = tmp_path / 'nested.zarr'
    zarr.open_group(str(root), mode='w', zarr_format=2)
    _write_multiscale_ome_zarr(root / 'sub')

    arrays, scale, _translate = open_ome_zarr(str(root))

    assert len(arrays) == 2
    assert scale == [2.0, 2.0]


def test_open_ome_zarr_bare_pyramid_no_metadata(tmp_path):
    """A group of level arrays without OME multiscales metadata falls
    back to sorted child arrays — and _find_multiscales must not recurse
    into those arrays (which would fancy-index them and crash)."""
    import zarr

    root = tmp_path / 'bare.zarr'
    group = zarr.open_group(str(root), mode='w', zarr_format=2)
    for level, size in enumerate((16, 8, 4)):
        arr = group.create_array(
            name=str(level), shape=(size, size), chunks=(4, 4), dtype='u1'
        )
        # values exceed the array's axis length: a stray fancy-index
        # (the recursion bug) would raise BoundsCheckError
        arr[:] = 200

    arrays, scale, translate = open_ome_zarr(str(root))

    assert [a.shape for a in arrays] == [(16, 16), (8, 8), (4, 4)]
    assert scale is None
    assert translate is None


def test_open_ome_zarr_empty_datasets_fallback(tmp_path):
    """multiscales present but with an empty datasets list still falls
    back to the group's child arrays."""
    import zarr

    root = tmp_path / 'empty_ds.zarr'
    group = zarr.open_group(str(root), mode='w', zarr_format=2)
    for level, size in enumerate((16, 8)):
        arr = group.create_array(
            name=str(level), shape=(size, size), chunks=(4, 4), dtype='u1'
        )
        arr[:] = 1
    group.attrs['multiscales'] = [{'datasets': []}]

    arrays, _scale, _translate = open_ome_zarr(str(root))

    assert [a.shape for a in arrays] == [(16, 16), (8, 8)]


def test_mandelbulb_rgb_dataset_uint16(tmp_path):
    """maxiter >= 256 drives a uint16 store dtype; chunk bytes must match
    the declared array dtype (no serialization size mismatch)."""
    dataset = mandelbulb_rgb_dataset(max_levels=2, tilesize=8, maxiter=300)
    arrays = dataset['arrays']
    assert arrays[0].dtype == np.dtype('<u2')
    chunk = np.asarray(arrays[-1][:])  # would raise on a byte-size mismatch
    assert chunk.shape[-1] == 3
    assert chunk.dtype == np.dtype('<u2')
