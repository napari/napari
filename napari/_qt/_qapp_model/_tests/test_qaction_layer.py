import numpy as np
import numpy.testing as npt
import pytest

from napari._qt._qapp_model.qactions._layer import (
    _copy_affine_to_clipboard,
    _copy_rotate_to_clipboard,
    _copy_scale_to_clipboard,
    _copy_shear_to_clipboard,
    _copy_spatial_to_clipboard,
    _copy_translate_to_clipboard,
    _paste_spatial_from_clipboard,
)
from napari.components import LayerList
from napari.layers.base._test_util_sample_layer import SampleLayer
from napari.utils.transforms import Affine


@pytest.fixture()
def layer_list():
    layer_1 = SampleLayer(
        data=np.empty((10, 10)),
        scale=(2, 3),
        translate=(1, 1),
        rotate=90,
        name='l1',
        affine=Affine(scale=(0.5, 0.5), translate=(1, 2), rotate=45),
        shear=[1],
    )
    layer_2 = SampleLayer(
        data=np.empty((10, 10)),
        scale=(1, 1),
        translate=(0, 0),
        rotate=0,
        name='l2',
        affine=Affine(),
        shear=[0],
    )
    layer_3 = SampleLayer(
        data=np.empty((10, 10)),
        scale=(1, 1),
        translate=(0, 0),
        rotate=0,
        name='l3',
        affine=Affine(),
        shear=[0],
    )

    ll = LayerList([layer_1, layer_2, layer_3])
    ll.selection = {layer_2}
    return ll


@pytest.mark.usefixtures('qtbot')
def test_copy_scale_to_clipboard(layer_list):
    _copy_scale_to_clipboard(layer_list['l1'])
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_equal(layer_list['l2'].scale, (2, 3))
    npt.assert_array_equal(layer_list['l3'].scale, (1, 1))
    npt.assert_array_equal(layer_list['l2'].translate, (0, 0))


@pytest.mark.usefixtures('qtbot')
def test_copy_translate_to_clipboard(layer_list):
    _copy_translate_to_clipboard(layer_list['l1'])
    npt.assert_array_equal(layer_list['l2'].translate, (0, 0))
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_equal(layer_list['l2'].translate, (1, 1))
    npt.assert_array_equal(layer_list['l3'].translate, (0, 0))
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))


@pytest.mark.usefixtures('qtbot')
def test_copy_rotate_to_clipboard(layer_list):
    _copy_rotate_to_clipboard(layer_list['l1'])
    npt.assert_array_almost_equal(layer_list['l2'].rotate, ([1, 0], [0, 1]))
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_almost_equal(layer_list['l2'].rotate, ([0, -1], [1, 0]))
    npt.assert_array_almost_equal(layer_list['l3'].rotate, ([1, 0], [0, 1]))
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))


@pytest.mark.usefixtures('qtbot')
def test_copy_affine_to_clipboard(layer_list):
    _copy_affine_to_clipboard(layer_list['l1'])
    npt.assert_array_almost_equal(
        layer_list['l2'].affine.linear_matrix, Affine().linear_matrix
    )
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_almost_equal(
        layer_list['l2'].affine.linear_matrix,
        layer_list['l1'].affine.linear_matrix,
    )
    npt.assert_array_almost_equal(
        layer_list['l3'].affine.linear_matrix, Affine().linear_matrix
    )
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))


@pytest.mark.usefixtures('qtbot')
def test_copy_shear_to_clipboard(layer_list):
    _copy_shear_to_clipboard(layer_list['l1'])
    npt.assert_array_almost_equal(layer_list['l2'].shear, (0,))
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_almost_equal(layer_list['l2'].shear, (1,))
    npt.assert_array_almost_equal(layer_list['l3'].shear, (0,))
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))


@pytest.mark.usefixtures('qtbot')
def test_copy_spatial_to_clipboard(layer_list):
    _copy_spatial_to_clipboard(layer_list['l1'])
    npt.assert_array_equal(layer_list['l2'].scale, (1, 1))
    _paste_spatial_from_clipboard(layer_list)
    npt.assert_array_equal(layer_list['l2'].scale, (2, 3))
    npt.assert_array_equal(layer_list['l2'].translate, (1, 1))
    npt.assert_array_almost_equal(layer_list['l2'].rotate, ([0, -1], [1, 0]))
    npt.assert_array_almost_equal(
        layer_list['l2'].affine.linear_matrix,
        layer_list['l1'].affine.linear_matrix,
    )
    npt.assert_array_equal(layer_list['l3'].scale, (1, 1))
