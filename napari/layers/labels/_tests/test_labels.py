import itertools

import numpy as np
import pytest
import xarray as xr
from numpy.core.numerictypes import issubdtype
from numpy.testing import assert_array_almost_equal, assert_raises
from skimage import data

from napari._tests.utils import check_layer_world_data_extent
from napari.layers import Labels
from napari.utils import Colormap
from napari.utils.colormaps import low_discrepancy_image


def test_random_labels():
    """Test instantiating Labels layer with random 2D data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape)
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True


def test_all_zeros_labels():
    """Test instantiating Labels layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape)
    assert layer._data_view.shape == shape[-2:]


def test_3D_labels():
    """Test instantiating Labels layer with random 3D data."""
    shape = (6, 10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape)
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True

    layer._slice_dims(ndisplay=3)
    assert layer._ndisplay == 3
    assert layer.editable is False
    assert layer.mode == 'pan_zoom'


def test_float_labels():
    """Test instantiating labels layer with floats"""
    np.random.seed(0)
    data = np.random.uniform(0, 20, size=(10, 10))
    with pytest.raises(TypeError):
        Labels(data)

    data0 = np.random.uniform(20, size=(20, 20))
    data1 = data0[::2, ::2].astype(np.int32)
    data = [data0, data1]
    with pytest.raises(TypeError):
        Labels(data)


def test_bool_labels():
    """Test instantiating labels layer with bools"""
    data = np.zeros((10, 10), dtype=bool)
    layer = Labels(data)
    assert issubdtype(layer.data.dtype, np.integer)

    data0 = np.zeros((20, 20), dtype=bool)
    data1 = data0[::2, ::2].astype(np.int32)
    data = [data0, data1]
    layer = Labels(data)
    assert all(issubdtype(d.dtype, np.integer) for d in layer.data)


def test_changing_labels():
    """Test changing Labels data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    shape_c = (10, 10)
    np.random.seed(0)
    data_a = np.random.randint(20, size=shape_a)
    data_b = np.random.randint(20, size=shape_b)
    layer = Labels(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape_b)
    assert layer._data_view.shape == shape_b[-2:]

    data_c = np.zeros(shape_c, dtype=bool)
    layer.data = data_c
    assert np.issubdtype(layer.data.dtype, np.integer)

    data_c = data_c.astype(np.float32)
    with pytest.raises(TypeError):
        layer.data = data_c


def test_changing_labels_dims():
    """Test changing Labels data including dimensionality."""
    shape_a = (10, 15)
    shape_b = (20, 12, 6)
    np.random.seed(0)
    data_a = np.random.randint(20, size=shape_a)
    data_b = np.random.randint(20, size=shape_b)
    layer = Labels(data_a)

    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    np.testing.assert_array_equal(layer.extent.data[1] + 1, shape_b)
    assert layer._data_view.shape == shape_b[-2:]


def test_changing_modes():
    """Test changing modes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'fill'
    assert layer.mode == 'fill'
    assert layer.interactive is False

    layer.mode = 'paint'
    assert layer.mode == 'paint'
    assert layer.interactive is False

    layer.mode = 'pick'
    assert layer.mode == 'pick'
    assert layer.interactive is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'paint'
    assert layer.mode == 'paint'
    layer.editable = False
    assert layer.mode == 'pan_zoom'
    assert layer.editable is False


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.name == 'Labels'

    layer = Labels(data, name='random')
    assert layer.name == 'random'

    layer.name = 'lbls'
    assert layer.name == 'lbls'


def test_visiblity():
    """Test setting layer visibility."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Labels(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.opacity == 0.7

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Labels(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Labels(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_seed():
    """Test setting seed."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.seed == 0.5

    layer.seed = 0.9
    assert layer.seed == 0.9

    layer = Labels(data, seed=0.7)
    assert layer.seed == 0.7

    # ensure setting seed triggers
    # recalculation of _all_vals
    _all_vals_07 = layer._all_vals.copy()
    layer.seed = 0.4
    _all_vals_04 = layer._all_vals.copy()
    assert_raises(
        AssertionError, assert_array_almost_equal, _all_vals_04, _all_vals_07
    )


def test_num_colors():
    """Test setting number of colors in colormap."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.num_colors == 50

    layer.num_colors = 80
    assert layer.num_colors == 80

    layer = Labels(data, num_colors=60)
    assert layer.num_colors == 60


def test_properties():
    """Test adding labels with properties."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))

    layer = Labels(data)
    assert isinstance(layer.properties, dict)
    assert len(layer.properties) == 0

    properties = {'class': ['Background'] + [f'Class {i}' for i in range(20)]}
    label_index = {i: i for i in range(len(properties['class']))}
    layer = Labels(data, properties=properties)
    assert isinstance(layer.properties, dict)
    assert layer.properties == properties
    assert layer._label_index == label_index

    current_label = layer.get_value((0, 0))
    layer_message = layer.get_status((0, 0))
    assert layer_message.endswith(f'Class {current_label - 1}')

    properties = {'class': ['Background']}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer_message.endswith("[No Properties]")

    properties = {'class': ['Background', 'Class 12'], 'index': [0, 12]}
    label_index = {0: 0, 12: 1}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message.endswith('Class 12')


def test_default_properties_assignment():
    """Test that the default properties value can be assigned to properties
    see https://github.com/napari/napari/issues/2477
    """
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))

    layer = Labels(data)
    layer.properties = {}
    assert layer.properties == {}


def test_multiscale_properties():
    """Test adding labels with multiscale properties."""
    np.random.seed(0)
    data0 = np.random.randint(20, size=(10, 15))
    data1 = data0[::2, ::2]
    data = [data0, data1]

    layer = Labels(data)
    assert isinstance(layer.properties, dict)
    assert len(layer.properties) == 0

    properties = {'class': ['Background'] + [f'Class {i}' for i in range(20)]}
    label_index = {i: i for i in range(len(properties['class']))}
    layer = Labels(data, properties=properties)
    assert isinstance(layer.properties, dict)
    assert layer.properties == properties
    assert layer._label_index == label_index

    current_label = layer.get_value((0, 0))[1]
    layer_message = layer.get_status((0, 0))
    assert layer_message.endswith(f'Class {current_label - 1}')

    properties = {'class': ['Background']}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer_message.endswith("[No Properties]")

    properties = {'class': ['Background', 'Class 12'], 'index': [0, 12]}
    label_index = {0: 0, 12: 1}
    layer = Labels(data, properties=properties)
    layer_message = layer.get_status((0, 0))
    assert layer._label_index == label_index
    assert layer_message.endswith('Class 12')


def test_colormap():
    """Test colormap."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert isinstance(layer.colormap, Colormap)
    assert layer.colormap.name == 'label_colormap'

    layer.new_colormap()
    assert isinstance(layer.colormap, Colormap)
    assert layer.colormap.name == 'label_colormap'


def test_custom_color_dict():
    """Test custom color dict."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(
        data, color={2: 'white', 4: 'red', 8: 'blue', 16: 'red', 32: 'blue'}
    )

    # test with custom color dict
    assert type(layer.get_color(2)) == np.ndarray
    assert type(layer.get_color(1)) == np.ndarray
    assert (layer.get_color(2) == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert (layer.get_color(4) == layer.get_color(16)).all()
    assert (layer.get_color(8) == layer.get_color(32)).all()

    # test disable custom color dict
    # should not initialize as white since we are using random.seed
    layer.color_mode = 'auto'
    assert not (layer.get_color(1) == np.array([1.0, 1.0, 1.0, 1.0])).all()


def test_add_colors():
    """Test adding new colors"""
    data = np.random.randint(20, size=(40, 40))
    layer = Labels(data)
    assert len(layer._all_vals) == layer.num_colors

    layer.selected_label = 51
    assert len(layer._all_vals) == 52

    layer.show_selected_label = True
    layer.selected_label = 53
    assert len(layer._all_vals) == 54


def test_metadata():
    """Test setting labels metadata."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.metadata == {}

    layer = Labels(data, metadata={'unit': 'cm'})
    assert layer.metadata == {'unit': 'cm'}


def test_brush_size():
    """Test changing brush size."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.brush_size == 10

    layer.brush_size = 20
    assert layer.brush_size == 20


def test_contiguous():
    """Test changing contiguous."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.contiguous is True

    layer.contiguous = False
    assert layer.contiguous is False


def test_n_edit_dimensions():
    """Test changing the number of editable dimensions."""
    np.random.seed(0)
    data = np.random.randint(20, size=(5, 10, 15))
    layer = Labels(data)
    layer.n_edit_dimensions = 2
    with pytest.warns(FutureWarning):
        assert layer.n_dimensional is False
    layer.n_edit_dimensions = 3
    with pytest.warns(FutureWarning):
        assert layer.n_dimensional is True


@pytest.mark.parametrize(
    "input_data, expected_data_view",
    [
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
                    [0, 0, 1, 0, 1, 5, 0, 5, 0, 0],
                    [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
        ),
        (
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 4, 4, 4],
                ],
                dtype=np.int_,
            ),
            np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4, 4, 4, 4],
                    [3, 3, 3, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
                ],
                dtype=np.int_,
            ),
        ),
        (
            5 * np.ones((9, 10), dtype=np.uint32),
            np.zeros((9, 10), dtype=np.uint32),
        ),
    ],
)
def test_contour(input_data, expected_data_view):
    """Test changing contour."""
    layer = Labels(input_data)
    assert layer.contour == 0
    np.testing.assert_array_equal(layer.data, input_data)

    np.testing.assert_array_equal(
        layer._raw_to_displayed(input_data), layer._data_view
    )
    data_view_before_contour = layer._data_view.copy()

    layer.contour = 1
    assert layer.contour == 1

    # Check `layer.data` didn't change
    np.testing.assert_array_equal(layer.data, input_data)

    # Check what is returned in the view of the data
    np.testing.assert_array_equal(
        layer._data_view,
        np.where(
            expected_data_view > 0,
            low_discrepancy_image(expected_data_view),
            0,
        ),
    )

    # Check the view of the data changed after setting the contour
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(
            data_view_before_contour, layer._data_view
        )

    layer.contour = 0
    assert layer.contour == 0

    # Check it's in the same state as before setting the contour
    np.testing.assert_array_equal(
        layer._raw_to_displayed(input_data), layer._data_view
    )


def test_selecting_label():
    """Test selecting label."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.selected_label == 1
    assert (layer._selected_color == layer.get_color(1)).all

    layer.selected_label = 1
    assert layer.selected_label == 1
    assert len(layer._selected_color) == 4


def test_label_color():
    """Test getting label color."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    col = layer.get_color(0)
    assert col is None

    col = layer.get_color(1)
    assert len(col) == 4


def test_show_selected_label():
    """Test color of labels when filtering to selected labels"""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    original_color = layer.get_color(1)

    layer.show_selected_label = True
    original_background_color = layer.get_color(layer._background_label)
    none_color = layer.get_color(None)
    layer.selected_label = 1

    # color of selected label has not changed
    assert np.allclose(layer.get_color(layer.selected_label), original_color)

    current_background_color = layer.get_color(layer._background_label)
    # color of background is background color
    assert current_background_color == original_background_color

    # color of all others is none color
    other_labels = np.unique(layer.data)[2:]
    other_colors = np.array(
        list(map(lambda x: layer.get_color(x), other_labels))
    )
    assert np.allclose(other_colors, none_color)


def test_paint():
    """Test painting labels with different square brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 1
    layer = Labels(data)

    with pytest.warns(FutureWarning):
        layer.brush_shape = 'square'
    assert np.unique(layer.data[:5, :5]) == 1
    assert np.unique(layer.data[5:10, 5:10]) == 1

    layer.brush_size = 9
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[:5, :5]) == 2
    assert np.unique(layer.data[5:10, 5:10]) == 1

    layer.brush_size = 10
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[:6, :6]) == 2
    assert np.unique(layer.data[6:10, 6:10]) == 1

    layer.brush_size = 19
    layer.paint([0, 0], 2)
    assert np.unique(layer.data[:5, :5]) == 2
    assert np.unique(layer.data[5:10, 5:10]) == 2


def test_paint_with_preserve_labels():
    """Test painting labels with square brush while preserving existing labels."""
    data = np.zeros((15, 10), dtype=np.uint32)
    data[:3, :3] = 1
    layer = Labels(data)
    with pytest.warns(FutureWarning):
        layer.brush_shape = 'square'
    layer.preserve_labels = True
    assert np.unique(layer.data[:3, :3]) == 1

    layer.brush_size = 9
    layer.paint([0, 0], 2)

    assert np.unique(layer.data[3:5, 0:5]) == 2
    assert np.unique(layer.data[0:5, 3:5]) == 2
    assert np.unique(layer.data[:3, :3]) == 1


@pytest.mark.parametrize(
    "brush_shape, expected_sum",
    [("circle", [41, 137, 137, 41, 349]), ("square", [36, 144, 169, 36, 400])],
)
def test_paint_2d(brush_shape, expected_sum):
    """Test painting labels with circle/square brush."""
    data = np.zeros((40, 40), dtype=np.uint32)
    layer = Labels(data)
    layer.brush_size = 12
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape
    layer.mode = 'paint'
    layer.paint((0, 0), 3)

    layer.brush_size = 12
    layer.paint((15, 8), 4)

    layer.brush_size = 13
    layer.paint((30.2, 7.8), 5)

    layer.brush_size = 12
    layer.paint((39, 39), 6)

    layer.brush_size = 20
    layer.paint((15, 27), 7)

    assert np.sum(layer.data[:8, :8] == 3) == expected_sum[0]
    assert np.sum(layer.data[9:22, 2:15] == 4) == expected_sum[1]
    assert np.sum(layer.data[24:37, 2:15] == 5) == expected_sum[2]
    assert np.sum(layer.data[33:, 33:] == 6) == expected_sum[3]
    assert np.sum(layer.data[5:26, 17:38] == 7) == expected_sum[4]


@pytest.mark.timeout(1)
@pytest.mark.parametrize(
    "brush_shape, expected_sum",
    [("circle", 411), ("square", 432)],
)
def test_paint_2d_xarray(brush_shape, expected_sum):
    """Test the memory usage of painting an xarray indirectly via timeout."""
    data = xr.DataArray(np.zeros((3, 3, 1024, 1024), dtype=np.uint32))

    layer = Labels(data)
    layer.brush_size = 12
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape
    layer.mode = 'paint'
    layer.paint((1, 1, 512, 512), 3)
    assert isinstance(layer.data, xr.DataArray)
    assert layer.data.sum() == expected_sum


@pytest.mark.parametrize(
    "brush_shape, expected_sum",
    [("circle", [137, 1189, 1103]), ("square", [144, 1728, 1548])],
)
def test_paint_3d(brush_shape, expected_sum):
    """Test painting labels with circle/square brush on 3D image."""
    data = np.zeros((30, 40, 40), dtype=np.uint32)
    layer = Labels(data)
    layer.brush_size = 12
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape
    layer.mode = 'paint'

    # Paint in 2D
    layer.paint((10, 10, 10), 3)

    # Paint in 3D
    layer.n_edit_dimensions = 3
    layer.paint((10, 25, 10), 4)

    # Paint in 3D, preserve labels
    layer.n_edit_dimensions = 3
    layer.preserve_labels = True
    layer.paint((10, 15, 15), 5)

    assert np.sum(layer.data[4:17, 4:17, 4:17] == 3) == expected_sum[0]
    assert np.sum(layer.data[4:17, 19:32, 4:17] == 4) == expected_sum[1]
    assert np.sum(layer.data[4:17, 9:32, 9:32] == 5) == expected_sum[2]


def test_fill():
    """Test filling labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 2
    data[:5, :5] = 1
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5]) == 1
    assert np.unique(layer.data[5:10, 5:10]) == 2

    layer.fill([0, 0], 3)
    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[5:10, 5:10]) == 2


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    value = layer.get_value((0, 0))
    assert value == data[0, 0]


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    msg = layer.get_status((0, 0))
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.randint(20, size=(30, 30))
    layer = Labels(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_world_data_extent():
    """Test extent after applying transforms."""
    np.random.seed(0)
    shape = (6, 10, 15)
    data = np.random.randint(20, size=(shape))
    layer = Labels(data)
    extent = np.array(((0,) * 3, np.subtract(shape, 1)))
    check_layer_world_data_extent(layer, extent, (3, 1, 1), (10, 20, 5))


@pytest.mark.parametrize(
    'brush_shape, brush_size, mode, selected_label, preserve_labels, n_dimensional',
    list(
        itertools.product(
            ['square', 'circle'],
            list(range(1, 22, 5)),
            ['fill', 'erase', 'paint'],
            [1, 20, 100],
            [True, False],
            [True, False],
        )
    ),
)
def test_undo_redo(
    brush_shape,
    brush_size,
    mode,
    selected_label,
    preserve_labels,
    n_dimensional,
):
    blobs = data.binary_blobs(length=64, volume_fraction=0.3, n_dim=3)
    layer = Labels(blobs)
    data_history = [blobs.copy()]
    with pytest.warns(FutureWarning):
        layer.brush_shape = brush_shape
    layer.brush_size = brush_size
    layer.mode = mode
    layer.selected_label = selected_label
    layer.preserve_labels = preserve_labels
    layer.n_edit_dimensions = 3 if n_dimensional else 2
    coord = np.random.random((3,)) * (np.array(blobs.shape) - 1)
    while layer.data[tuple(coord.astype(int))] == 0 and np.any(layer.data):
        coord = np.random.random((3,)) * (np.array(blobs.shape) - 1)
    if layer.mode == 'fill':
        layer.fill(coord, layer.selected_label)
    if layer.mode == 'erase':
        layer.paint(coord, 0)
    if layer.mode == 'paint':
        layer.paint(coord, layer.selected_label)
    data_history.append(np.copy(layer.data))
    layer.undo()
    np.testing.assert_array_equal(layer.data, data_history[0])
    layer.redo()
    np.testing.assert_array_equal(layer.data, data_history[1])


def test_ndim_fill():
    test_array = np.zeros((5, 5, 5, 5), dtype=int)

    test_array[:, 1:3, 1:3, 1:3] = 1

    layer = Labels(test_array)
    layer.n_edit_dimensions = 3

    layer.fill((0, 1, 1, 1), 2)

    np.testing.assert_equal(layer.data[0, 1:3, 1:3, 1:3], 2)
    np.testing.assert_equal(layer.data[1, 1:3, 1:3, 1:3], 1)

    layer.n_edit_dimensions = 4

    layer.fill((1, 1, 1, 1), 3)

    np.testing.assert_equal(layer.data[0, 1:3, 1:3, 1:3], 2)
    np.testing.assert_equal(layer.data[1:, 1:3, 1:3, 1:3], 3)


def test_ndim_paint():
    test_array = np.zeros((5, 6, 7, 8), dtype=int)
    layer = Labels(test_array)
    layer.n_edit_dimensions = 3
    with pytest.warns(FutureWarning):
        layer.brush_shape = 'circle'
    layer.brush_size = 2  # equivalent to 18-connected 3D neighborhood
    layer.paint((1, 1, 1, 1), 1)

    assert np.sum(layer.data) == 19  # 18 + center
    assert not np.any(layer.data[0]) and not np.any(layer.data[2:])

    layer.n_edit_dimensions = 2  # 3x3 square
    layer._dims_order = [1, 2, 0, 3]
    layer.paint((4, 5, 6, 7), 8)
    assert len(np.flatnonzero(layer.data == 8)) == 4  # 2D square is in corner
    np.testing.assert_array_equal(
        test_array[:, 5, 6, :],
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 8, 8],
                [0, 0, 0, 0, 0, 0, 8, 8],
            ]
        ),
    )


def test_switching_display_func():
    label_data = np.random.randint(2 ** 25, 2 ** 25 + 5, size=(50, 50))
    layer = Labels(label_data)
    assert layer._color_lookup_func == layer._lookup_with_low_discrepancy_image

    label_data = np.random.randint(0, 5, size=(50, 50))
    layer = Labels(label_data)
    assert layer._color_lookup_func == layer._lookup_with_index
