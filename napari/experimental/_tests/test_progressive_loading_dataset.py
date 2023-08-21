import dask
import pytest
import zarr

from napari.experimental._progressive_loading_datasets import (
    idr0044A,
    idr0051A,
    idr0075A,
    luethi_zenodo_7144919,
    mandelbrot_dataset,
    openorganelle_mouse_kidney_em,
    openorganelle_mouse_kidney_labels,
)

# list of (working) dataset loaders
dataset_loaders = [
    openorganelle_mouse_kidney_labels,
    openorganelle_mouse_kidney_em,
    luethi_zenodo_7144919,
    mandelbrot_dataset,
]

# dataset loaders which currently fail
dataset_loaders_fail = [
    idr0044A,
    idr0075A,
    idr0051A,
]


@pytest.mark.slow
@pytest.mark.parametrize("dataset_loader", dataset_loaders)
def test_datasets(dataset_loader):
    """Test to ensure that datasets are functional. Most of these are
    time consuming and therefore should only be run infrequently.
    """
    large_image = dataset_loader()
    sample_array = large_image['arrays'][0]
    assert isinstance(sample_array, (dask.array.Array, zarr.Array))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.parametrize("dataset_loader", dataset_loaders_fail)
def test_failing_datasets(dataset_loader):
    """Temporary placeholder for datasets that currently fail during creation"""
    dataset_loader()


@pytest.mark.parametrize("level", [2, 4])
def test_mandelbrot_dataset(level):
    large_image = mandelbrot_dataset(max_levels=level)
    multiscale_img = large_image["arrays"]

    assert isinstance(multiscale_img[0], zarr.Array)
    assert len(multiscale_img) == level
