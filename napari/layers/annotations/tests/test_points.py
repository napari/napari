import numpy as np
from copy import copy
from xml.etree.ElementTree import Element
from napari.layers import Points


def test_empty_points():
    pts = Points()
    assert pts.data.shape == (0, 2)


def test_random_points():
    """Test instantiating Points layer with random 2D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 10
    assert len(layer.selected_data) == 0


def test_integer_points():
    """Test instantiating Points layer with integer data."""
    shape = (10, 2)
    np.random.seed(0)
    data = np.random.randint(20, size=(10, 2))
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 10


def test_negative_points():
    """Test instantiating Points layer with negative data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) - 10
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 10


def test_empty_points():
    """Test instantiating Points layer with empty array."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 0


def test_3D_points():
    """Test instantiating Points layer with random 3D data."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 10


def test_4D_points():
    """Test instantiating Points layer with random 4D data."""
    shape = (10, 4)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert np.all(layer.data == data)
    assert layer.ndim == shape[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 10


def test_changing_points():
    """Test changing Points data."""
    shape_a = (10, 2)
    shape_b = (20, 2)
    np.random.seed(0)
    data_a = 20 * np.random.random(shape_a)
    data_b = 20 * np.random.random(shape_b)
    layer = Points(data_a)
    layer.data = data_b
    assert np.all(layer.data == data_b)
    assert layer.ndim == shape_b[1]
    assert layer._data_view.ndim == 2
    assert len(layer.data) == 20


def test_selecting_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    layer.selected_data = [0, 1]
    assert layer.selected_data == [0, 1]


def test_adding_points():
    """Test adding Points data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert len(layer.data) == 10

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 11
    assert np.all(layer.data[10] == coord)


def test_adding_points_to_empty():
    """Test adding Points data to empty."""
    shape = (0, 2)
    data = np.empty(shape)
    layer = Points(data)
    assert len(layer.data) == 0

    coord = [20, 20]
    layer.add(coord)
    assert len(layer.data) == 1
    assert np.all(layer.data[0] == coord)


def test_removing_selected_points():
    """Test selecting points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == shape[0]

    # Select two points and remove them
    layer.selected_data = [0, 3]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 2
    assert len(layer.selected_data) == 0
    keep = [1, 2] + list(range(4, 10))
    assert np.all(layer.data == data[keep])

    # Select another point and remove it
    layer.selected_data = [4]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 3


def test_move():
    """Test moving points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    unmoved = copy(data)
    layer = Points(data)

    # Move one point relative to an initial drag start location
    layer._move([0], [0, 0])
    layer._move([0], [10, 10])
    layer._drag_start = None
    assert np.all(layer.data[0] == unmoved[0] + [10, 10])
    assert np.all(layer.data[1:] == unmoved[1:])

    # Move two points relative to an initial drag start location
    layer._move([1, 2], [2, 2])
    layer._move([1, 2], np.add([2, 2], [-3, 4]))
    assert np.all(layer.data[1:2] == unmoved[1:2] + [-3, 4])


def test_changing_modes():
    """Test changing modes."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive == True

    layer.mode = 'add'
    assert layer.mode == 'add'
    assert layer.interactive == False

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.interactive == False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive == True


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.name == 'Points'

    layer = Points(data, name='random')
    assert layer.name == 'random'

    layer.name = 'pts'
    assert layer.name == 'pts'


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.visible == True

    layer.visible = False
    assert layer.visible == False

    layer = Points(data, visible=False)
    assert layer.visible == False

    layer.visible = True
    assert layer.visible == True


def test_opacity():
    """Test setting layer opacity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.opacity == 1.0

    layer.opacity = 0.5
    assert layer.opacity == 0.5

    layer = Points(data, opacity=0.6)
    assert layer.opacity == 0.6

    layer.opacity = 0.3
    assert layer.opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 2))
    layer = Points(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Points(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_symbol():
    """Test setting symbol."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.symbol == 'disc'

    layer.symbol = 'cross'
    assert layer.symbol == 'cross'

    layer = Points(data, symbol='star')
    assert layer.symbol == 'star'


def test_edge_width():
    """Test setting edge width."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.edge_width == 1

    layer.edge_width = 2
    assert layer.edge_width == 2

    layer = Points(data, edge_width=3)
    assert layer.edge_width == 3


def test_n_dimensional():
    """Test setting n_dimensional flag for 2D and 4D data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.n_dimensional == False

    layer.n_dimensional = True
    assert layer.n_dimensional == True

    layer = Points(data, n_dimensional=True)
    assert layer.n_dimensional == True

    shape = (10, 4)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.n_dimensional == False

    layer.n_dimensional = True
    assert layer.n_dimensional == True

    layer = Points(data, n_dimensional=True)
    assert layer.n_dimensional == True


def test_edge_color():
    """Test setting edge color."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.edge_color == 'black'
    assert len(layer.edge_colors) == shape[0]
    assert np.all([col == 'black' for col in layer.edge_colors])

    # With no data selected chaning edge color has no effect
    layer.edge_color = 'blue'
    assert layer.edge_color == 'blue'
    assert np.all([col == 'black' for col in layer.edge_colors])

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.edge_color == 'black'
    layer.edge_color = 'green'
    assert np.all([col == 'green' for col in layer.edge_colors[:2]])
    assert np.all([col == 'black' for col in layer.edge_colors[2:]])

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = []
    layer.edge_color = 'blue'
    layer.add(coord)
    assert len(layer.edge_colors) == shape[0] + 1
    assert np.all([col == 'green' for col in layer.edge_colors[:2]])
    assert np.all([col == 'black' for col in layer.edge_colors[2:10]])
    assert np.all(layer.edge_colors[10] == 'blue')

    # Instantiate with custom edge color
    layer = Points(data, edge_color='red')
    assert layer.edge_color == 'red'

    # Instantiate with custom edge color list
    col_list = ['red', 'green'] * 5
    layer = Points(data, edge_color=col_list)
    assert layer.edge_color == 'black'
    assert layer.edge_colors == col_list

    # Add new point and test its color
    coord = [18, 18]
    layer.edge_color = 'blue'
    layer.add(coord)
    assert len(layer.edge_colors) == shape[0] + 1
    assert layer.edge_colors == col_list + ['blue']

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_colors) == shape[0] - 1
    assert layer.edge_colors == [col_list[1]] + col_list[3:] + ['blue']


def test_face_color():
    """Test setting face color."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.face_color == 'white'
    assert len(layer.face_colors) == shape[0]
    assert np.all([col == 'white' for col in layer.face_colors])

    # With no data selected chaning face color has no effect
    layer.face_color = 'blue'
    assert layer.face_color == 'blue'
    assert np.all([col == 'white' for col in layer.face_colors])

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.face_color == 'white'
    layer.face_color = 'green'
    assert np.all([col == 'green' for col in layer.face_colors[:2]])
    assert np.all([col == 'white' for col in layer.face_colors[2:]])

    # Add new point and test its color
    coord = [18, 18]
    layer.selected_data = []
    layer.face_color = 'blue'
    layer.add(coord)
    assert len(layer.face_colors) == shape[0] + 1
    assert np.all([col == 'green' for col in layer.face_colors[:2]])
    assert np.all([col == 'white' for col in layer.face_colors[2:10]])
    assert np.all(layer.face_colors[10] == 'blue')

    # Instantiate with custom face color
    layer = Points(data, face_color='red')
    assert layer.face_color == 'red'

    # Instantiate with custom face color list
    col_list = ['red', 'green'] * 5
    layer = Points(data, face_color=col_list)
    assert layer.face_color == 'white'
    assert layer.face_colors == col_list

    # Add new point and test its color
    coord = [18, 18]
    layer.face_color = 'blue'
    layer.add(coord)
    assert len(layer.face_colors) == shape[0] + 1
    assert layer.face_colors == col_list + ['blue']

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.face_colors) == shape[0] - 1
    assert layer.face_colors == [col_list[1]] + col_list[3:] + ['blue']


def test_size():
    """Test setting size with scalar."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.unique(layer.sizes)[0] == 10

    # Add a new point, it should get current size
    coord = [17, 17]
    layer.add(coord)
    assert layer.sizes.shape == (11, 2)
    assert np.unique(layer.sizes)[0] == 10

    # Setting size affects newly added points not current points
    layer.size = 20
    assert layer.size == 20
    assert layer.sizes.shape == (11, 2)
    assert np.unique(layer.sizes)[0] == 10

    # Add new point, should have new size
    coord = [18, 18]
    layer.add(coord)
    assert layer.sizes.shape == (12, 2)
    assert np.unique(layer.sizes[:11])[0] == 10
    assert np.all(layer.sizes[11] == [20, 20])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.size == 10
    layer.size = 16
    assert layer.sizes.shape == (12, 2)
    assert np.unique(layer.sizes[2:11])[0] == 10
    assert np.unique(layer.sizes[:2])[0] == 16

    # Select data and size changes
    layer.selected_data = [11]
    assert layer.size == 20

    # Create new layer with new size data
    layer = Points(data, size=15)
    assert layer.size == 15
    assert layer.sizes.shape == shape
    assert np.unique(layer.sizes)[0] == 15


def test_size_with_arrays():
    """Test setting size with arrays."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    sizes = 5 * np.random.random(shape)
    layer.sizes = sizes
    assert np.all(layer.sizes == sizes)

    # Test broadcasting of sizes
    sizes = [5, 5]
    layer.sizes = sizes
    assert np.all(layer.sizes[0] == sizes)

    # Create new layer with new size array data
    sizes = 5 * np.random.random(shape)
    layer = Points(data, size=sizes)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.all(layer.sizes == sizes)

    # Create new layer with new size array data
    sizes = [5, 5]
    layer = Points(data, size=sizes)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.all(layer.sizes[0] == sizes)

    # Add new point, should have new size
    coord = [18, 18]
    layer.size = 13
    layer.add(coord)
    assert layer.sizes.shape == (11, 2)
    assert np.unique(layer.sizes[:10])[0] == 5
    assert np.all(layer.sizes[10] == [13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.size == 5
    layer.size = 16
    assert layer.sizes.shape == (11, 2)
    assert np.unique(layer.sizes[2:10])[0] == 5
    assert np.unique(layer.sizes[:2])[0] == 16

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == 9
    assert len(layer.sizes) == 9
    assert np.all(layer.sizes[0] == [16, 16])
    assert np.all(layer.sizes[1] == [5, 5])


def test_size_with_3D_arrays():
    """Test setting size with 3D arrays."""
    shape = (10, 3)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[:2, 0] = 0
    layer = Points(data)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.unique(layer.sizes)[0] == 10

    sizes = 5 * np.random.random(shape)
    layer.sizes = sizes
    assert np.all(layer.sizes == sizes)

    # Test broadcasting of sizes
    sizes = [1, 5, 5]
    layer.sizes = sizes
    assert np.all(layer.sizes[0] == sizes)

    # Create new layer with new size array data
    sizes = 5 * np.random.random(shape)
    layer = Points(data, size=sizes)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.all(layer.sizes == sizes)

    # Create new layer with new size array data
    sizes = [1, 5, 5]
    layer = Points(data, size=sizes)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.all(layer.sizes[0] == sizes)

    # Add new point, should have new size in last dim only
    coord = [4, 18, 18]
    layer.size = 13
    layer.add(coord)
    assert layer.sizes.shape == (11, 3)
    assert np.unique(layer.sizes[:10, 1:])[0] == 5
    assert np.all(layer.sizes[10] == [1, 13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.size == 5
    layer.size = 16
    assert layer.sizes.shape == (11, 3)
    assert np.unique(layer.sizes[2:10, 1:])[0] == 5
    assert np.all(layer.sizes[0] == [16, 16, 16])

    # Create new 3D layer with new 2D points size data
    sizes = [0, 5, 5]
    layer = Points(data, size=sizes)
    assert layer.size == 10
    assert layer.sizes.shape == shape
    assert np.all(layer.sizes[0] == sizes)

    # Add new point, should have new size only in last 2 dimensions
    coord = [4, 18, 18]
    layer.size = 13
    layer.add(coord)
    assert layer.sizes.shape == (11, 3)
    assert np.all(layer.sizes[10] == [0, 13, 13])

    # Select data and change size
    layer.selected_data = [0, 1]
    assert layer.size == 5
    layer.size = 16
    assert layer.sizes.shape == (11, 3)
    assert np.unique(layer.sizes[2:10, 1:])[0] == 5
    assert np.all(layer.sizes[0] == [0, 16, 16])


def test_interaction_box():
    """Test the creation of the interaction box."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    assert layer._selected_box == None

    layer.selected_data = [0]
    assert len(layer._selected_box) == 4

    layer.selected_data = [0, 1]
    assert len(layer._selected_box) == 4

    layer.selected_data = []
    assert layer._selected_box == None


def test_copy_and_paste():
    """Test copying and pasting selected points."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two points selected adds to clipboard and data
    layer.selected_data = [0, 1]
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard.keys()) > 0
    assert len(layer.data) == shape[0] + 2
    assert np.all(layer.data[:2] == layer.data[-2:])

    # Pasting again adds two more points to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.all(layer.data[:2] == layer.data[-2:])

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = []
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == 9

    layer.data = layer.data + 20
    value = layer.get_value()
    assert value == None


def test_message():
    """Test converting value and coords to message."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1] = [0, 0]
    layer = Points(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[0] = [0, 0]
    data[-1] = [20, 20]
    layer = Points(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_xml_list():
    """Test the xml generation."""
    shape = (10, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Points(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == shape[0]
    assert np.all([type(x) == Element for x in xml])
