import numpy as np
from xml.etree.ElementTree import Element
from vispy.color import Colormap
from napari.layers import Labels
import collections


def test_random_labels():
    """Test instantiating Labels layer with random 2D data."""
    shape = (10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer.dims.range == [(0, m, 1) for m in shape]
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True


def test_all_zeros_labels():
    """Test instantiating Labels layer with all zeros data."""
    shape = (10, 15)
    data = np.zeros(shape, dtype=int)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer._data_view.shape == shape[-2:]


def test_3D_labels():
    """Test instantiating Labels layer with random 3D data."""
    shape = (6, 10, 15)
    np.random.seed(0)
    data = np.random.randint(20, size=shape)
    layer = Labels(data)
    assert np.all(layer.data == data)
    assert layer.ndim == len(shape)
    assert layer.shape == shape
    assert layer._data_view.shape == shape[-2:]
    assert layer.editable is True

    layer.dims.ndisplay = 3
    assert layer.dims.ndisplay == 3
    assert layer.editable is False
    assert layer.mode == 'pan_zoom'


def test_changing_labels():
    """Test changing Labels data."""
    shape_a = (10, 15)
    shape_b = (20, 12)
    np.random.seed(0)
    data_a = np.random.randint(20, size=shape_a)
    data_b = np.random.randint(20, size=shape_b)
    layer = Labels(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == len(shape_b)
    assert layer.shape == shape_b
    assert layer.dims.range == [(0, m, 1) for m in shape_b]
    assert layer._data_view.shape == shape_b[-2:]


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
    assert layer.shape == shape_b
    assert layer.dims.range == [(0, m, 1) for m in shape_b]
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

    layer.mode = 'picker'
    assert layer.mode == 'picker'
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
    """Test setting layer visiblity."""
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


def test_colormap():
    """Test colormap."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert type(layer.colormap) == tuple
    assert layer.colormap[0] == 'random'
    assert type(layer.colormap[1]) == Colormap

    layer.new_colormap()
    assert type(layer.colormap) == tuple
    assert layer.colormap[0] == 'random'
    assert type(layer.colormap[1]) == Colormap


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


def test_n_dimensional():
    """Test changing n_dimensional."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.n_dimensional is False

    layer.n_dimensional = True
    assert layer.n_dimensional is True


def test_selecting_label():
    """Test selecting label."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    assert layer.selected_label == 0
    assert layer._selected_color is None

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


def test_paint():
    """Test painting labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 1
    layer = Labels(data)
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


def test_fill():
    """Test filling labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    data[:10, :10] = 2
    data[:5, :5] = 1
    layer = Labels(data)
    assert np.unique(layer.data[:5, :5]) == 1
    assert np.unique(layer.data[5:10, 5:10]) == 2

    layer.fill([0, 0], 1, 3)
    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[5:10, 5:10]) == 2


def test_value():
    """Test getting the value of the data at the current coordinates."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == data[0, 0]


def test_message():
    """Test converting value and coords to message."""
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 15))
    layer = Labels(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    np.random.seed(0)
    data = np.random.randint(20, size=(30, 30))
    layer = Labels(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    np.random.seed(0)
    data = np.random.randint(20, size=(30, 30))
    layer = Labels(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == 1
    assert type(xml[0]) == Element


def test_mouse_move():
    """Test painting labels with different brush sizes."""
    np.random.seed(0)
    data = np.random.randint(20, size=(20, 20))
    layer = Labels(data)
    layer.brush_size = 10
    layer.mode = 'paint'
    layer.selected_label = 3
    layer._last_cursor_coord = (0, 0)
    layer.coordinates = (19, 19)
    Event = collections.namedtuple('Event', 'is_dragging')
    event = Event(is_dragging=True)
    layer.on_mouse_move(event)

    assert np.unique(layer.data[:5, :5]) == 3
    assert np.unique(layer.data[-5:, -5:]) == 3
