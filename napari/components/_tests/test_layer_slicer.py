import time

import numpy as np
import pytest

from napari.components import Dims
from napari.components._layer_slicer import _LayerSlicer
from napari.layers.base.base import _LayerSliceRequest, _LayerSliceResponse
from napari.layers.image.image import AsyncImage

# pytest napari/components/_tests/test_layer_slicer.py -svv


@pytest.fixture()
def async_layer():
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.random(shape)
    layer = AsyncImage(data)
    return layer


@pytest.fixture()
def viewer_slice_request(layer_slice_request, async_layer):
    """[Layer, _LayerSliceRequest]"""

    requests = {
        async_layer: layer_slice_request,
    }
    return requests


@pytest.fixture()
def viewer_slice_response(layer_slice_response, async_layer):
    """[Layer, _LayerSliceResponse]
    slice response should be a future"""
    responses = {
        async_layer: layer_slice_response,
    }
    return responses


@pytest.fixture()
def layer_slice_request():
    return _LayerSliceRequest(
        data=None,
        data_to_world=None,
        ndim=None,
        ndisplay=None,
        point=None,
        dims_order=None,
        dims_displayed=None,
        dims_not_displayed=None,
        multiscale=None,
        corner_pixels=None,
        round_index=None,
    )


@pytest.fixture()
def layer_slice_response(layer_slice_request):
    return _LayerSliceResponse(
        request=layer_slice_request,
        data=None,
        data_to_world=None,
    )


def test_force_sync():
    layer_slicer = _LayerSlicer()
    assert not layer_slicer._force_sync

    with layer_slicer.force_sync() as manager:
        assert not manager
        assert layer_slicer._force_sync

    assert not layer_slicer._force_sync


def test_slice_layers_async(async_layer):
    layer_slicer = _LayerSlicer()
    dims = Dims()
    assert dims.ndim == 2
    slice_reponse = layer_slicer.slice_layers_async(
        layers=[async_layer],
        dims=dims,
    )
    assert slice_reponse


def test_slice_layers(viewer_slice_request):
    """requires
    Layer._get_slice
    """
    layer_slicer = _LayerSlicer()
    slice_reponse = layer_slicer._slice_layers(
        requests=viewer_slice_request,
    )
    assert isinstance(slice_reponse, dict)


def test_on_slice_done(layer_slice_response):
    """TODO to test this properly, it needs to be done at a higher level to
    check result on the Layer."""

    # test no errors are raised for a simple submit
    layer_slicer = _LayerSlicer()
    with layer_slicer._executor as executor:
        task = executor.submit(tuple, (1, 2))
        response = layer_slicer._on_slice_done(
            task=task,
        )
        assert response is None
        assert task.done()

    # test cancellation of task
    layer_slicer = _LayerSlicer()
    with layer_slicer._executor as executor:
        task = executor.submit(time.sleep, 0.1)
        task.cancel()
        response = layer_slicer._on_slice_done(
            task=task,
        )
        assert response is None
        assert task.done()


def test_executor():
    layer_slicer = _LayerSlicer()
    with layer_slicer._executor as executor:
        task1 = executor.submit(time.sleep, 0.1)
        task2 = executor.submit(time.sleep, 0.2)
        task1.result()
    assert not task1.running()
    assert not task2.running()
