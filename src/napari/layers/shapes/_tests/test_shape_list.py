import numpy as np
import numpy.testing as npt
import pytest

from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_models import Path, Polygon, Rectangle


@pytest.fixture
def shape_li():
    """Fixture for ShapeList."""
    shape1 = Rectangle(np.array([[0, 0], [10, 10]]))
    shape2 = Rectangle(np.array([[10, 0], [0, 10]]))
    shape3 = Rectangle(np.array([[20, 20], [30, 30]]))
    shape4 = Rectangle(np.array([[-10, -10], [0, 0]]))
    res = ShapeList()
    res.add([shape1, shape2, shape3, shape4])
    return res


@pytest.fixture
def shape_li_3d():
    """Fixture for ShapeList."""
    shape1 = Rectangle(
        np.array([[0, 0, 0], [0, 10, 0], [0, 10, 10], [0, 0, 10]])
    )
    shape2 = Rectangle(
        np.array([[1, 10, 0], [1, 10, 10], [1, 0, 10], [1, 0, 0]])
    )
    shape3 = Rectangle(
        np.array([[0, 20, 20], [0, 20, 30], [0, 30, 30], [0, 30, 20]])
    )
    shape4 = Rectangle(
        np.array([[1, -10, -10], [1, -10, 0], [1, 0, 0], [1, 0, -10]])
    )
    res = ShapeList()
    res.add([shape1, shape2, shape3, shape4])
    return res


def test_empty_shape_list():
    """Test instantiating empty ShapeList."""
    shape_list = ShapeList()
    assert len(shape_list.shapes) == 0


def test_adding_to_shape_list():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)
    assert len(shape_list.shapes) == 1
    assert shape_list.shapes[0] == shape


def test_reset_bounding_box_rotation():
    """Test if rotating shape resets bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, np.array([[[-0.5, -0.5]], [[10.5, 10.5]]])
    )
    shape_list.rotate(0, 45, (5, 5))
    p = 5 * np.sqrt(2) + 0.5
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[5 - p, 5 - p], [5 + p, 5 + p]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_reset_bounding_box_shift():
    """Test if shifting shape resets bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )
    shape_list.shift(0, np.array([5, 5]))
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[4.5, 4.5], [15.5, 15.5]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_reset_bounding_box_scale():
    """Test if scaling shape resets the bounding box."""
    shape = Rectangle(np.array([[0, 0], [10, 10]]))
    shape_list = ShapeList()
    shape_list.add(shape)
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )
    shape_list.scale(0, 2, (5, 5))
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-5.5, -5.5], [15.5, 15.5]])
    )
    npt.assert_array_almost_equal(
        shape_list._bounding_boxes, shape.bounding_box[:, np.newaxis, :]
    )


def test_shape_list_outline():
    """Test ShapeList outline method."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)

    # Check passing an int
    outline_by_index = shape_list.outline(0)
    assert isinstance(outline_by_index, tuple)

    # Check passing a list
    outline_by_index_list = shape_list.outline([0])
    assert isinstance(outline_by_index_list, tuple)

    # Check return value for `int` and `list` are the same
    for value_by_idx, value_by_idx_list in zip(
        outline_by_index, outline_by_index_list, strict=False
    ):
        assert np.array_equal(value_by_idx, value_by_idx_list)

    # Check passing a `numpy.int_` (`numpy.int32/64` depending on platform)
    outline_by_index_np = shape_list.outline(np.int_(0))
    assert isinstance(outline_by_index_np, tuple)

    # Check return value for `int` and `numpy.int_` are the same
    for value_by_idx, value_by_idx_np in zip(
        outline_by_index, outline_by_index_np, strict=False
    ):
        assert np.array_equal(value_by_idx, value_by_idx_np)


def test_shape_list_outline_two_shapes():
    shape1 = Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    shape2 = Polygon([[20, 20], [20, 30], [30, 30], [30, 20]])
    shape_list = ShapeList()
    shape_list.add([shape1, shape2])

    # check if the outline contains triangle with vertex of number 16

    triangles = shape_list.outline([0, 1])[2]
    assert np.any(triangles == 16)


def test_nD_shapes():
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    shape_list = ShapeList()
    data = 20 * np.random.random((6, 3))
    data[:, 0] = 0
    shape_a = Polygon(data)
    shape_list.add(shape_a)

    data = 20 * np.random.random((6, 3))
    data[:, 0] = 1
    shape_b = Path(data)
    shape_list.add(shape_b)

    assert len(shape_list.shapes) == 2
    assert shape_list.shapes[0] == shape_a
    assert shape_list.shapes[1] == shape_b

    assert shape_list._vertices.shape[1] == 2
    assert shape_list._mesh.vertices.shape[1] == 2

    shape_list.ndisplay = 3
    assert shape_list._vertices.shape[1] == 3
    assert shape_list._mesh.vertices.shape[1] == 3


@pytest.mark.parametrize('attribute', ['edge', 'face'])
def test_bad_color_array(attribute):
    """Test adding shapes to ShapeList."""
    np.random.seed(0)
    data = 20 * np.random.random((4, 2))
    shape = Rectangle(data)
    shape_list = ShapeList()

    shape_list.add(shape)

    # test setting color with a color array of the wrong shape
    bad_color_array = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    with pytest.raises(ValueError, match='must have shape'):
        setattr(shape_list, f'{attribute}_color', bad_color_array)


def test_inside():
    shape1 = Polygon(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]]))
    shape2 = Polygon(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]))
    shape3 = Polygon(np.array([[2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 0, 1]]))

    shape_list = ShapeList()
    shape_list.add([shape1, shape2, shape3])
    shape_list.slice_key = (1,)
    assert shape_list.inside((0.5, 0.5)) == 1


def test_visible_shapes_4d():
    """Test _visible_shapes with 4D data like those from OME-Zarr/OMERO."""
    shape1 = Polygon(
        np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 50],
                [0, 0, 50, 50],
                [0, 0, 50, 10],
            ]
        )
    )
    shape2 = Polygon(
        np.array(
            [
                [0, 1, 10, 10],
                [0, 1, 10, 50],
                [0, 1, 50, 50],
                [0, 1, 50, 10],
            ]
        )
    )
    shape3 = Polygon(
        np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 50],
                [0, 1, 50, 50],
                [0, 1, 50, 10],
            ]
        )
    )

    shape_list = ShapeList()
    # set slice_key first to avoid empty array broadcasting error
    shape_list.slice_key = np.array([0, 0])
    shape_list.add([shape1, shape2, shape3])

    # at (0,0) - should show shape1 and shape3
    shape_list.slice_key = np.array([0, 0])
    visible = shape_list._visible_shapes
    assert len(visible) == 2
    visible_shapes = [v[1] for v in visible]
    assert shape1 in visible_shapes
    assert shape3 in visible_shapes

    # at (0,1) - should show shape2 and shape3
    shape_list.slice_key = np.array([0, 1])
    visible = shape_list._visible_shapes
    assert len(visible) == 2
    visible_shapes = [v[1] for v in visible]
    assert shape2 in visible_shapes
    assert shape3 in visible_shapes


def test_remove_shape(shape_li):
    """Test removing shapes from ShapeList."""
    initial_length = len(shape_li.shapes)
    assert initial_length == 4
    assert shape_li._vertices_index.shape[0] == 5

    # Remove a shape by index
    shape_li.remove(1)
    assert len(shape_li.shapes) == initial_length - 1
    assert shape_li._vertices_index.shape[0] == 4


def test_edit_shape_simple(shape_li):
    """Test editing shapes in ShapeList."""
    initial_shape = shape_li.shapes[0]
    assert isinstance(initial_shape, Rectangle)

    # Edit the first shape
    shape_li.edit(1, np.array([[5, 5], [15, 15]]))

    # Check if the shape has been updated
    assert len(shape_li.shapes) == 4
    npt.assert_array_equal(
        shape_li.shapes[1].data,
        np.array([[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]]),
    )


def test_edit_shape_triangle(shape_li):
    """Test editing a shape to a triangle in ShapeList."""
    initial_shape = shape_li.shapes[0]
    assert isinstance(initial_shape, Rectangle)

    # Edit the first shape to a triangle
    shape_li.edit(1, np.array([[5, 5], [15, 5], [10, 15]]), new_type=Polygon)

    # Check if the shape has been updated
    assert len(shape_li.shapes) == 4
    npt.assert_array_equal(
        shape_li.shapes[1].data,
        np.array([[5.0, 5.0], [15.0, 5.0], [10.0, 15.0]]),
    )

    assert shape_li._mesh.triangles.shape[0] == 40


def test_edit_shape_pentagon(shape_li):
    """Test editing a shape to a polygon in ShapeList."""
    initial_shape = shape_li.shapes[0]
    assert isinstance(initial_shape, Rectangle)

    # Edit the first shape to a polygon
    shape_li.edit(
        1,
        np.array([[0, 0], [10, 0], [10, 10], [8, 15], [0, 10]]),
        new_type=Polygon,
    )

    # Check if the shape has been updated
    assert len(shape_li.shapes) == 4
    npt.assert_array_equal(
        shape_li.shapes[1].data,
        np.array([[0, 0], [10, 0], [10, 10], [8, 15], [0, 10]]),
    )
    assert shape_li._mesh.triangles.shape[0] == 43


@pytest.mark.parametrize(
    'new_color',
    [
        np.array([1, 0, 0, 1]),
        np.array([[1, 0, 0, 1]]),
        np.array([[1, 0, 0, 1]] * 4),
    ],
)
def test_update_face_color(shape_li, new_color):
    """Test updating face color of shapes in ShapeList."""
    # Initial face color
    initial_edge_color = shape_li.edge_color
    assert initial_edge_color.shape == (4, 4)

    # Update face color
    expected_color = np.array([[1, 0, 0, 1]] * 4)  # Red color
    shape_li.update_face_colors(range(4), new_color)

    # Check if the face color has been updated
    npt.assert_array_equal(shape_li.face_color, expected_color)
    assert (
        np.count_nonzero(
            np.all(
                shape_li._mesh.displayed_triangles_colors == expected_color[0],
                axis=1,
            )
        )
        == 8
    )

    # Check if the edge color remains unchanged
    assert np.all(shape_li.edge_color == initial_edge_color)


@pytest.mark.parametrize(
    'new_color',
    [
        np.array([1, 0, 0, 1]),
        np.array([[1, 0, 0, 1]]),
        np.array([[1, 0, 0, 1]] * 4),
    ],
)
def test_update_edge_color(shape_li, new_color):
    """Test updating edge color of shapes in ShapeList."""
    # Initial edge color
    initial_face_color = shape_li.face_color
    assert initial_face_color.shape == (4, 4)

    # Update edge color
    expected_color = np.array([[1, 0, 0, 1]] * 4)
    shape_li.update_edge_colors(range(4), new_color)
    # Check if the edge color has been updated
    npt.assert_array_equal(shape_li.edge_color, expected_color)
    assert (
        np.count_nonzero(
            np.all(
                shape_li._mesh.displayed_triangles_colors == expected_color[0],
                axis=1,
            )
        )
        == 32
    )

    # Check if the face color remains unchanged
    assert np.all(shape_li.face_color == initial_face_color)


def test_multi_layer_data(shape_li_3d):
    shape_li_3d.slice_key = (1,)
    assert shape_li_3d._mesh.displayed_triangles_colors.shape[0] == 20


LAYER_COUNT = 16


@pytest.fixture
def multi_z_rectangles() -> list[Rectangle]:
    width = 6
    edge = width - 2
    return [  # create 256 boxes in a grid
        Rectangle(
            np.array(
                [
                    [z, y * width, x * width],
                    [z, y * width + edge, x * width],
                    [z, y * width + edge, x * width + edge],
                    [z, y * width, x * width + edge],
                ]
            )
        )
        for x in range(4)
        for y in range(4)
        for z in range(LAYER_COUNT)
    ]


@pytest.fixture
def simple_rectangle():
    return Rectangle([(0, 0), (6, 0), (6, 6), (0, 6)])


@pytest.fixture
def triangles_slice(simple_rectangle):
    triang = np.concatenate(
        [
            simple_rectangle._face_triangles,
            simple_rectangle._edge_triangles
            + simple_rectangle.face_vertices_count,
        ]
    )

    return np.concatenate(
        [
            triang + i * simple_rectangle.vertices_count
            for i in range(0, 16 * LAYER_COUNT, LAYER_COUNT)
        ]
    )


@pytest.mark.parametrize('slice_', list(range(LAYER_COUNT)))
def test_proper_shape_position(
    multi_z_rectangles, slice_, triangles_slice, simple_rectangle
):
    sl = ShapeList(multi_z_rectangles)
    sl.ndisplay = 2
    sl.slice_key = (slice_,)
    assert sl._mesh.displayed_triangles_colors.shape[0] == 160
    npt.assert_array_equal(
        sl._mesh.displayed_triangles,
        triangles_slice + slice_ * simple_rectangle.vertices_count,
    )
