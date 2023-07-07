import numpy as np
import pytest
# from _mandelbrot_vizarr import add_progressive_loading_image
# from napari.experimental._progressive_loading import (
#     add_progressive_loading_image,
# )
from numpy.testing import assert_array_equal, assert_raises
import dask
import napari
import zarr
from napari.experimental import _progressive_loading
from napari.experimental._progressive_loading import get_chunk
from napari.experimental._progressive_loading_datasets import (
    openorganelle_mouse_kidney_labels,
    openorganelle_mouse_kidney_em,
    idr0044A,
    idr0075A,
    idr0051A,
    luethi_zenodo_7144919,
    mandelbrot_dataset,

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
    assert isinstance(sample_array, dask.array.Array) or isinstance(sample_array, zarr.Array)

@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.parametrize("dataset_loader", dataset_loaders_fail)
def test_failing_datasets(dataset_loader): 
    """Temporary placeholder for datasets that currently fail during creation"""
    dataset_loader()


