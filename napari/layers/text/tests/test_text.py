import numpy as np
from copy import deepcopy
from xml.etree.ElementTree import Element
from napari.layers import Text


def test_random_text():
    """Test instantiating Text layer with random 2D data."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert np.all(layer.data[0] == coords)
    assert layer.text == text
    assert layer.data[1] == text
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text) == shape[0]
    assert len(layer.text_coords) == shape[0]
    assert len(layer.selected_data) == 0


def test_integer_text():
    """Test instantiating Text layer with integer coordinates."""
    shape = (10, 2)
    np.random.seed(0)
    coords = np.random.randint(20, size=shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert np.all(layer.data[0] == coords)
    assert layer.text == text
    assert layer.data[1] == text
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text) == shape[0]
    assert len(layer.text_coords) == shape[0]
    assert len(layer.selected_data) == 0


def test_negative_text():
    """Test instantiating Text layer with negative data."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape) - 10
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text_coords) == shape[0]


def test_empty_text():
    """Test instantiating Text layer with empty array."""
    txt = Text()
    assert txt.text_coords.shape == (0, 2)
    assert txt.text == []

    shape = (0, 2)
    coords = np.empty(shape)
    text = []
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text_coords) == 0
    assert len(layer.text) == 0


def test_3D_text():
    """Test instantiating Text layer with random 3D data."""
    shape = (10, 3)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert np.all(layer.data[0] == coords)
    assert layer.text == text
    assert layer.data[1] == text
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text) == shape[0]
    assert len(layer.text_coords) == shape[0]
    assert len(layer.selected_data) == 0


def test_4D_text():
    """Test instantiating Text layer with random 4D data."""
    shape = (10, 4)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert np.all(layer.text_coords == coords)
    assert np.all(layer.data[0] == coords)
    assert layer.text == text
    assert layer.data[1] == text
    assert layer.ndim == shape[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text) == shape[0]
    assert len(layer.text_coords) == shape[0]
    assert len(layer.selected_data) == 0


def test_changing_text():
    """Test changing Text data."""
    shape_a = (10, 2)
    shape_b = (20, 2)
    np.random.seed(0)
    coords_a = 20 * np.random.random(shape_a)
    coords_b = 20 * np.random.random(shape_b)
    text_a = shape_a[0] * ['text_a']
    text_b = shape_b[0] * ['text_b']
    data_a = (coords_a, text_a)
    data_b = (coords_b, text_b)
    layer = Text(data_a)
    layer.data = data_b
    assert np.all(layer.text_coords == coords_b)
    assert layer.ndim == shape_b[1]
    assert layer._text_coords_view.ndim == 2
    assert len(layer.text) == shape_b[0]


def test_selecting_text():
    """Test selecting text."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    layer.selected_data = [0, 1]
    assert layer.selected_data == [0, 1]


def test_adding_text():
    """Test adding Text data."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert len(layer.text_coords) == shape[0]

    coord = [20, 20]
    layer.add(coord)
    n_text = shape[0] + 1
    last_index = n_text - 1
    assert len(layer.text_coords) == n_text
    assert layer.text[last_index] == 'edit'
    assert np.all(layer.text_coords[last_index] == coord)


def test_adding_points_to_empty():
    """Test adding Text data to empty."""
    shape = (0, 2)
    coords = np.empty(shape)
    text = []
    data = (coords, text)
    layer = Text(data)
    assert len(layer.text_coords) == 0

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.text_coords) == 1
    assert np.all(layer.text_coords[0] == coord)
    assert layer.text[0] == 'edit'


def test_removing_selected_text():
    """Test selecting text and removing them"""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.text_coords) == shape[0]

    # Select two points and remove them
    layer.selected_data = [0, 3]
    layer.remove_selected()
    assert len(layer.text_coords) == shape[0] - 2
    assert len(layer.text) == shape[0] - 2
    assert len(layer.selected_data) == 0
    keep = [1, 2] + list(range(4, 10))
    assert np.all(layer.text_coords == coords[keep])

    # Select another point and remove it
    layer.selected_data = [4]
    layer.remove_selected()
    assert len(layer.text_coords) == shape[0] - 3


def test_move():
    """Test moving points."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    unmoved = deepcopy(data)
    layer = Text(data)

    # Move one point relative to an initial drag start location
    layer._move([0], [0, 0])
    layer._move([0], [10, 10])
    layer._drag_start = None
    assert np.all(layer.text_coords[0] == unmoved[0][0] + [10, 10])
    assert np.all(layer.text_coords[1:] == unmoved[0][1:])
    assert np.all(layer.text == unmoved[1])

    # Move two points relative to an initial drag start location
    layer._move([1, 2], [2, 2])
    layer._move([1, 2], np.add([2, 2], [-3, 4]))
    assert np.all(layer.text_coords[1:2] == unmoved[0][1:2] + [-3, 4])


def test_changing_modes():
    """Test changing modes."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'add'
    assert layer.mode == 'add'
    assert layer.interactive is False

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.interactive is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True


def test_name():
    """Test setting layer name."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer.name == 'Text'

    layer = Text(data, name='random')
    assert layer.name == 'random'

    layer.name = 'txt'
    assert layer.name == 'txt'


def test_visiblity():
    """Test setting layer visiblity."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Text(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_opacity():
    """Test setting layer opacity."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Text(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Text(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_interaction_box():
    """Test the creation of the interaction box."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    assert layer._selected_box is None

    layer.selected_data = [0]
    assert len(layer._selected_box) == 4

    layer.selected_data = [0, 1]
    assert len(layer._selected_box) == 4

    layer.selected_data = []
    assert layer._selected_box is None


def test_copy_and_paste():
    """Test copying and pasting selected text."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.text_coords) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two points selected adds to clipboard and data
    layer.selected_data = [0, 1]
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard.keys()) > 0
    assert len(layer.text_coords) == shape[0] + 2
    assert np.all(layer.text_coords[:2] == layer.text_coords[-2:])

    # Pasting again adds two more points to data
    layer._paste_data()
    assert len(layer.text_coords) == shape[0] + 4
    assert np.all(layer.text_coords[:2] == layer.text_coords[-2:])

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = []
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.text_coords) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    coords[-1] = [0, 0]
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == 9

    new_coords = coords + 20
    layer.data = (new_coords, text)
    value = layer.get_value()
    assert value is None


def test_message():
    """Test converting value and coords to message."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    shape = (10, 2)
    np.random.seed(0)
    coords = 20 * np.random.random(shape)
    text = shape[0] * ['test']
    data = (coords, text)
    layer = Text(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == shape[0]
    assert np.all([type(x) == Element for x in xml])
