import numpy as np
from xml.etree.ElementTree import Element
from napari.layers import Shapes


def test_empty_shapes():
    shp = Shapes()
    assert shp.dims.ndim == 2


def test_rectangles():
    """Test instantiating Shapes layer with a random 2D rectangles."""
    # Test a single four corner rectangle
    shape = (1, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test multiple four corner rectangles
    shape = (10, 4, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test a single two corner rectangle, which gets converted into four
    # corner rectangle
    shape = (1, 2, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])

    # Test multiple two corner rectangles
    shape = (10, 2, 2)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_rectangles_roundtrip():
    """Test a full roundtrip with rectangles data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    new_layer = Shapes(layer.data)
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_integer_rectangle():
    """Test instantiating rectangles with integer data."""
    shape = (10, 2, 2)
    np.random.seed(1)
    data = np.random.randint(20, size=shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_negative_rectangle():
    """Test instantiating rectangles with negative data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape) - 10
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_empty_rectangle():
    """Test instantiating rectangles with empty data."""
    shape = (0, 0, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_3D_rectangles():
    """Test instantiating Shapes layer with 3D planar rectangles."""
    # Test a single four corner rectangle
    np.random.seed(0)
    planes = np.tile(np.arange(10).reshape((10, 1, 1)), (1, 4, 1))
    corners = np.random.uniform(0, 10, size=(10, 4, 2))
    data = np.concatenate((planes, corners), axis=2)
    layer = Shapes(data)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 3
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_ellipses():
    """Test instantiating Shapes layer with a random 2D ellipses."""
    # Test a single four corner ellipses
    shape = (1, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test multiple four corner ellipses
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test a single ellipse center radii, which gets converted into four
    # corner ellipse
    shape = (1, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == 1
    assert len(layer.data[0]) == 4
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])

    # Test multiple center radii ellipses
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == shape[0]
    assert np.all([len(ld) == 4 for ld in layer.data])
    assert layer.ndim == shape[2]
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_4D_ellispse():
    """Test instantiating Shapes layer with 4D planar ellipse."""
    # Test a single 4D ellipse
    np.random.seed(0)
    data = [
        [
            [3, 5, 108, 108],
            [3, 5, 108, 148],
            [3, 5, 148, 148],
            [3, 5, 148, 108],
        ]
    ]
    layer = Shapes(data, shape_type='ellipse')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 4
    assert np.all([s == 'ellipse' for s in layer.shape_type])


def test_ellipses_roundtrip():
    """Test a full roundtrip with ellipss data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='ellipse')
    new_layer = Shapes(layer.data, shape_type='ellipse')
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_lines():
    """Test instantiating Shapes layer with a random 2D lines."""
    # Test a single two end point line
    shape = (1, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])

    # Test multiple lines
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    assert layer.nshapes == shape[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == shape[2]
    assert np.all([s == 'line' for s in layer.shape_type])


def test_lines_roundtrip():
    """Test a full roundtrip with line data."""
    shape = (10, 2, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='line')
    new_layer = Shapes(layer.data, shape_type='line')
    assert np.all([nd == d for nd, d in zip(new_layer.data, layer.data)])


def test_paths():
    """Test instantiating Shapes layer with a random 2D paths."""
    # Test a single path with 6 points
    shape = (1, 6, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='path')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'path' for s in layer.shape_type])

    # Test multiple paths with different numbers of points
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='path')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == 'path' for s in layer.shape_type])


def test_paths_roundtrip():
    """Test a full roundtrip with path data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='path')
    new_layer = Shapes(layer.data, shape_type='path')
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )


def test_polygons():
    """Test instantiating Shapes layer with a random 2D polygons."""
    # Test a single polygon with 6 points
    shape = (1, 6, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data, shape_type='polygon')
    assert layer.nshapes == shape[0]
    assert np.all(layer.data[0] == data[0])
    assert layer.ndim == shape[2]
    assert np.all([s == 'polygon' for s in layer.shape_type])

    # Test multiple polygons with different numbers of points
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='polygon')
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == 'polygon' for s in layer.shape_type])


def test_polygon_roundtrip():
    """Test a full roundtrip with polygon data."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(10)
    ]
    layer = Shapes(data, shape_type='polygon')
    new_layer = Shapes(layer.data, shape_type='polygon')
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )


def test_mixed_shapes():
    """Test instantiating Shapes layer with a mix of random 2D shapes."""
    # Test multiple polygons with different numbers of points
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type=shape_type)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, shape_type)])

    # Test roundtrip with mixed data
    new_layer = Shapes(layer.data, shape_type=layer.shape_type)
    assert np.all(
        [np.all(nd == d) for nd, d in zip(new_layer.data, layer.data)]
    )
    assert np.all(
        [ns == s for ns, s in zip(new_layer.shape_type, layer.shape_type)]
    )


def test_changing_shapes():
    """Test changing Shapes data."""
    shape_a = (10, 4, 2)
    shape_b = (20, 4, 2)
    np.random.seed(0)
    data_a = 20 * np.random.random(shape_a)
    data_b = 20 * np.random.random(shape_b)
    layer = Shapes(data_a)
    assert layer.nshapes == shape_a[0]
    layer.data = data_b
    assert layer.nshapes == shape_b[0]
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data_b)])
    assert layer.ndim == shape_b[2]
    assert np.all([s == 'rectangle' for s in layer.shape_type])


def test_adding_shapes():
    """Test adding shapes."""
    # Start with polygons with different numbers of points
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ]
    # shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type='polygon')
    new_data = np.random.random((5, 4, 2))
    new_shape_type = ['rectangle'] * 3 + ['ellipse'] * 2
    layer.add(new_data, shape_type=new_shape_type)
    all_data = data + list(new_data)
    all_shape_type = ['polygon'] * 5 + new_shape_type
    assert layer.nshapes == len(all_data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, all_data)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, all_shape_type)])


def test_adding_shapes_to_empty():
    """Test adding shapes to empty."""
    data = np.empty((0, 0, 2))
    np.random.seed(0)
    layer = Shapes(np.empty((0, 0, 2)))
    assert len(layer.data) == 0

    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['path'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2

    layer.add(data, shape_type=shape_type)
    assert layer.nshapes == len(data)
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data)])
    assert layer.ndim == 2
    assert np.all([s == so for s, so in zip(layer.shape_type, shape_type)])


def test_selecting_shapes():
    """Test selecting shapes."""
    data = 20 * np.random.random((10, 4, 2))
    np.random.seed(0)
    layer = Shapes(data)
    layer.selected_data = [0, 1]
    assert layer.selected_data == [0, 1]

    layer.selected_data = [9]
    assert layer.selected_data == [9]

    layer.selected_data = []
    assert layer.selected_data == []


def test_removing_selected_shapes():
    """Test removing selected shapes."""
    np.random.seed(0)
    data = [
        20 * np.random.random((np.random.randint(2, 12), 2)) for i in range(5)
    ] + list(np.random.random((5, 4, 2)))
    shape_type = ['polygon'] * 5 + ['rectangle'] * 3 + ['ellipse'] * 2
    layer = Shapes(data, shape_type=shape_type)

    # With nothing selected no points should be removed
    layer.remove_selected()
    assert len(layer.data) == len(data)

    # Select three shapes and remove them
    layer.selected_data = [1, 7, 8]
    layer.remove_selected()
    keep = [0] + list(range(2, 7)) + [9]
    data_keep = [data[i] for i in keep]
    shape_type_keep = [shape_type[i] for i in keep]
    assert len(layer.data) == len(data_keep)
    assert len(layer.selected_data) == 0
    assert np.all([np.all(ld == d) for ld, d in zip(layer.data, data_keep)])
    assert layer.ndim == 2
    assert np.all(
        [s == so for s, so in zip(layer.shape_type, shape_type_keep)]
    )


def test_changing_modes():
    """Test changing modes."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True

    layer.mode = 'select'
    assert layer.mode == 'select'
    assert layer.interactive is False

    layer.mode = 'direct'
    assert layer.mode == 'direct'
    assert layer.interactive is False

    layer.mode = 'vertex_insert'
    assert layer.mode == 'vertex_insert'
    assert layer.interactive is False

    layer.mode = 'vertex_remove'
    assert layer.mode == 'vertex_remove'
    assert layer.interactive is False

    layer.mode = 'add_rectangle'
    assert layer.mode == 'add_rectangle'
    assert layer.interactive is False

    layer.mode = 'add_ellipse'
    assert layer.mode == 'add_ellipse'
    assert layer.interactive is False

    layer.mode = 'add_line'
    assert layer.mode == 'add_line'
    assert layer.interactive is False

    layer.mode = 'add_path'
    assert layer.mode == 'add_path'
    assert layer.interactive is False

    layer.mode = 'add_polygon'
    assert layer.mode == 'add_polygon'
    assert layer.interactive is False

    layer.mode = 'pan_zoom'
    assert layer.mode == 'pan_zoom'
    assert layer.interactive is True


def test_name():
    """Test setting layer name."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.name == 'Shapes'

    layer = Shapes(data, name='random')
    assert layer.name == 'random'

    layer.name = 'shps'
    assert layer.name == 'shps'


def test_visiblity():
    """Test setting layer visiblity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.visible is True

    layer.visible = False
    assert layer.visible is False

    layer = Shapes(data, visible=False)
    assert layer.visible is False

    layer.visible = True
    assert layer.visible is True


def test_current_opacity():
    """Test setting current layer opacity."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.current_opacity == 0.7

    layer.current_opacity = 0.5
    assert layer.current_opacity == 0.5

    layer = Shapes(data, opacity=0.6)
    assert layer.current_opacity == 0.6

    layer.current_opacity = 0.3
    assert layer.current_opacity == 0.3


def test_blending():
    """Test setting layer blending."""
    np.random.seed(0)
    data = 20 * np.random.random((10, 4, 2))
    layer = Shapes(data)
    assert layer.blending == 'translucent'

    layer.blending = 'additive'
    assert layer.blending == 'additive'

    layer = Shapes(data, blending='additive')
    assert layer.blending == 'additive'

    layer.blending = 'opaque'
    assert layer.blending == 'opaque'


def test_edge_color():
    """Test setting edge color."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.current_edge_color == 'black'
    assert len(layer.edge_color) == shape[0]
    assert layer.edge_color == ['black'] * shape[0]

    # With no data selected changing edge color has no effect
    layer.current_edge_color = 'blue'
    assert layer.current_edge_color == 'blue'
    assert layer.edge_color == ['black'] * shape[0]

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.current_edge_color == 'black'
    layer.current_edge_color = 'green'
    assert layer.edge_color == ['green'] * 2 + ['black'] * (shape[0] - 2)

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = []
    layer.current_edge_color = 'blue'
    layer.add(new_shape)
    assert len(layer.edge_color) == shape[0] + 1
    assert layer.edge_color == ['green'] * 2 + ['black'] * (shape[0] - 2) + [
        'blue'
    ]

    # Instantiate with custom edge color
    layer = Shapes(data, edge_color='red')
    assert layer.current_edge_color == 'red'

    # Instantiate with custom edge color list
    col_list = ['red', 'green'] * 5
    layer = Shapes(data, edge_color=col_list)
    assert layer.current_edge_color == 'black'
    assert layer.edge_color == col_list

    # Add new point and test its color
    layer.current_edge_color = 'blue'
    layer.add(new_shape)
    assert len(layer.edge_color) == shape[0] + 1
    assert layer.edge_color == col_list + ['blue']

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_color) == shape[0] - 1
    assert layer.edge_color == [col_list[1]] + col_list[3:] + ['blue']


def test_face_color():
    """Test setting face color."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.current_face_color == 'white'
    assert len(layer.face_color) == shape[0]
    assert layer.face_color == ['white'] * shape[0]

    # With no data selected changing face color has no effect
    layer.current_face_color = 'blue'
    assert layer.current_face_color == 'blue'
    assert layer.face_color == ['white'] * shape[0]

    # Select data and change face color of selection
    layer.selected_data = [0, 1]
    assert layer.current_face_color == 'white'
    layer.current_face_color = 'green'
    assert layer.face_color == ['green'] * 2 + ['white'] * (shape[0] - 2)

    # Add new shape and test its color
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = []
    layer.current_face_color = 'blue'
    layer.add(new_shape)
    assert len(layer.face_color) == shape[0] + 1
    assert layer.face_color == ['green'] * 2 + ['white'] * (shape[0] - 2) + [
        'blue'
    ]

    # Instantiate with custom face color
    layer = Shapes(data, face_color='red')
    assert layer.current_face_color == 'red'

    # Instantiate with custom face color list
    col_list = ['red', 'green'] * 5
    layer = Shapes(data, face_color=col_list)
    assert layer.current_face_color == 'white'
    assert layer.face_color == col_list

    # Add new point and test its color
    layer.current_face_color = 'blue'
    layer.add(new_shape)
    assert len(layer.face_color) == shape[0] + 1
    assert layer.face_color == col_list + ['blue']

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.face_color) == shape[0] - 1
    assert layer.face_color == [col_list[1]] + col_list[3:] + ['blue']


def test_edge_width():
    """Test setting edge width."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.current_edge_width == 1
    assert len(layer.edge_width) == shape[0]
    assert layer.edge_width == [1] * shape[0]

    # With no data selected changing edge width has no effect
    layer.current_edge_width = 2
    assert layer.current_edge_width == 2
    assert layer.edge_width == [1] * shape[0]

    # Select data and change edge color of selection
    layer.selected_data = [0, 1]
    assert layer.current_edge_width == 1
    layer.current_edge_width = 3
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2)

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = []
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert layer.edge_width == [3] * 2 + [1] * (shape[0] - 2) + [4]

    # Instantiate with custom edge width
    layer = Shapes(data, edge_width=5)
    assert layer.current_edge_width == 5

    # Instantiate with custom edge width list
    width_list = [2, 3] * 5
    layer = Shapes(data, edge_width=width_list)
    assert layer.current_edge_width == 1
    assert layer.edge_width == width_list

    # Add new shape and test its color
    layer.current_edge_width = 4
    layer.add(new_shape)
    assert len(layer.edge_width) == shape[0] + 1
    assert layer.edge_width == width_list + [4]

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.edge_width) == shape[0] - 1
    assert layer.edge_width == [width_list[1]] + width_list[3:] + [4]


def test_opacity():
    """Test setting opacity."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Check default opacity value of 0.7
    assert layer.current_opacity == 0.7
    assert len(layer.opacity) == shape[0]
    assert layer.opacity == [0.7] * shape[0]

    # With no data selected changing opacity has no effect
    layer.current_opacity = 1
    assert layer.current_opacity == 1
    assert layer.opacity == [0.7] * shape[0]

    # Select data and change opacity of selection
    layer.selected_data = [0, 1]
    assert layer.current_opacity == 0.7
    layer.current_opacity = 0.5
    assert layer.opacity == [0.5] * 2 + [0.7] * (shape[0] - 2)

    # Add new shape and test its width
    new_shape = np.random.random((1, 4, 2))
    layer.selected_data = []
    layer.current_opacity = 0.3
    layer.add(new_shape)
    assert layer.opacity == [0.5] * 2 + [0.7] * (shape[0] - 2) + [0.3]

    # Instantiate with custom opacity
    layer = Shapes(data, opacity=0.2)
    assert layer.current_opacity == 0.2

    # Instantiate with custom opacity list
    opacity_list = [0.1, 0.4] * 5
    layer = Shapes(data, opacity=opacity_list)
    assert layer.current_opacity == 0.7
    assert layer.opacity == opacity_list

    # Add new shape and test its opacity
    layer.current_opacity = 0.6
    layer.add(new_shape)
    assert len(layer.opacity) == shape[0] + 1
    assert layer.opacity == opacity_list + [0.6]

    # Check removing data adjusts opacity correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.opacity) == shape[0] - 1
    assert layer.opacity == [opacity_list[1]] + opacity_list[3:] + [0.6]


def test_z_index():
    """Test setting z-index during instantiation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer.z_index == [0] * shape[0]

    # Instantiate with custom z-index
    layer = Shapes(data, z_index=4)
    assert layer.z_index == [4] * shape[0]

    # Instantiate with custom z-index list
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Add new shape and its z-index
    new_shape = np.random.random((1, 4, 2))
    layer.add(new_shape)
    assert len(layer.z_index) == shape[0] + 1
    assert layer.z_index == z_index_list + [4]

    # Check removing data adjusts colors correctly
    layer.selected_data = [0, 2]
    layer.remove_selected()
    assert len(layer.data) == shape[0] - 1
    assert len(layer.z_index) == shape[0] - 1
    assert layer.z_index == [z_index_list[1]] + z_index_list[3:] + [4]


def test_move_to_front():
    """Test moving shapes to front."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = [0, 2]
    layer.move_to_front()
    assert layer.z_index == [4] + [z_index_list[1]] + [4] + z_index_list[3:]


def test_move_to_back():
    """Test moving shapes to back."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    z_index_list = [2, 3] * 5
    layer = Shapes(data, z_index=z_index_list)
    assert layer.z_index == z_index_list

    # Move selected shapes to front
    layer.selected_data = [0, 2]
    layer.move_to_back()
    assert layer.z_index == [1] + [z_index_list[1]] + [1] + z_index_list[3:]


def test_interaction_box():
    """Test the creation of the interaction box."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    assert layer._selected_box is None

    layer.selected_data = [0]
    assert len(layer._selected_box) == 10

    layer.selected_data = [0, 1]
    assert len(layer._selected_box) == 10

    layer.selected_data = []
    assert layer._selected_box is None


def test_copy_and_paste():
    """Test copying and pasting selected shapes."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    # Clipboard starts empty
    assert layer._clipboard == {}

    # Pasting empty clipboard doesn't change data
    layer._paste_data()
    assert len(layer.data) == 10

    # Copying with nothing selected leave clipboard empty
    layer._copy_data()
    assert layer._clipboard == {}

    # Copying and pasting with two shapes selected adds to clipboard and data
    layer.selected_data = [0, 1]
    layer._copy_data()
    layer._paste_data()
    assert len(layer._clipboard) == 2
    assert len(layer.data) == shape[0] + 2
    assert np.all(
        [np.all(a == b) for a, b in zip(layer.data[:2], layer.data[-2:])]
    )

    # Pasting again adds two more points to data
    layer._paste_data()
    assert len(layer.data) == shape[0] + 4
    assert np.all(
        [np.all(a == b) for a, b in zip(layer.data[:2], layer.data[-2:])]
    )

    # Unselecting everything and copying and pasting will empty the clipboard
    # and add no new data
    layer.selected_data = []
    layer._copy_data()
    layer._paste_data()
    assert layer._clipboard == {}
    assert len(layer.data) == shape[0] + 4


def test_value():
    """Test getting the value of the data at the current coordinates."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 10], [10, 0], [10, 10]]
    layer = Shapes(data)
    value = layer.get_value()
    assert layer.coordinates == (0, 0)
    assert value == (9, None)

    layer.mode = 'select'
    layer.selected_data = [9]
    value = layer.get_value()
    assert value == (9, 7)

    layer = Shapes(data + 5)
    value = layer.get_value()
    assert value == (None, None)


def test_message():
    """Test converting values and coords to message."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    msg = layer.get_message()
    assert type(msg) == str


def test_thumbnail():
    """Test the image thumbnail for square data."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    data[-1, :] = [[0, 0], [0, 20], [20, 0], [20, 20]]
    layer = Shapes(data)
    layer._update_thumbnail()
    assert layer.thumbnail.shape == layer._thumbnail_shape


def test_to_masks():
    """Test the mask generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    masks = layer.to_masks()
    assert masks.ndim == 3
    assert len(masks) == shape[0]

    masks = layer.to_masks(mask_shape=[20, 20])
    assert masks.shape == (shape[0], 20, 20)


def test_to_labels():
    """Test the labels generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    labels = layer.to_labels()
    assert labels.ndim == 2
    assert len(np.unique(labels)) <= 11

    labels = layer.to_labels(labels_shape=[20, 20])
    assert labels.shape == (20, 20)
    assert len(np.unique(labels)) <= 11


def test_to_labels_3D():
    """Test label generation for 3D data"""
    data = [
        [[0, 100, 100], [0, 100, 200], [0, 200, 200], [0, 200, 100]],
        [[1, 125, 125], [1, 125, 175], [1, 175, 175], [1, 175, 125]],
        [[2, 100, 100], [2, 100, 200], [2, 200, 200], [2, 200, 100]],
    ]
    labels_shape = (3, 300, 300)
    layer = Shapes(np.array(data), shape_type='polygon')
    labels = layer.to_labels(labels_shape=labels_shape)
    assert np.all(labels.shape == labels_shape)
    assert np.all(np.unique(labels) == [0, 1, 2, 3])


def test_xml_list():
    """Test the xml generation."""
    shape = (10, 4, 2)
    np.random.seed(0)
    data = 20 * np.random.random(shape)
    layer = Shapes(data)
    xml = layer.to_xml_list()
    assert type(xml) == list
    assert len(xml) == shape[0]
    assert np.all([type(x) == Element for x in xml])
