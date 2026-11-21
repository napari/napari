import sys

import numpy as np
import numpy.testing as npt
import pytest
from vispy.geometry import PolygonData

from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.shapes._shapes_utils import (
    triangulate_face,
    triangulate_face_triangle,
)
from napari.settings import get_settings
from napari.settings._experimental import (
    TriangulationBackend,
)

BETTER_TRIANGULATION = 'triangle' in sys.modules or 'bermuda' in sys.modules

BACKEND_TO_MODULE = {
    TriangulationBackend.triangle: 'triangle',
    TriangulationBackend.bermuda: 'bermuda',
    TriangulationBackend.partsegcore: 'PartSegCore_compiled_backend',
    TriangulationBackend.pure_python: 'napari',
}


@pytest.fixture(autouse=True, params=BACKEND_TO_MODULE)
def triangulation_backend(request):
    """Fixture to switch between triangulation backends."""
    settings = get_settings()
    prev = settings.experimental.triangulation_backend
    if BACKEND_TO_MODULE[request.param] not in sys.modules:
        pytest.importorskip(str(request.param))
    settings.experimental.triangulation_backend = request.param
    yield request.param
    settings.experimental.triangulation_backend = prev


def test_rectangle1():
    """Test creating Rectangle by four corners."""
    np.random.seed(0)
    data = np.array([(10, 10), (20, 10), (20, 20), (10, 20)], dtype=np.float32)
    shape = Rectangle(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_rectangle2():
    """Test creating Rectangle by upper left and bottom right."""
    # If given two corners, representation will be expanded to four
    data = np.array([[-10, -10], [20, 20]], dtype=np.float32)
    shape = Rectangle(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_rectangle_bounding_box():
    """Test that the bounding box is correctly updated based on edge width."""
    data = [[10, 10], [20, 20]]
    shape = Rectangle(data)
    npt.assert_array_equal(
        shape.bounding_box, np.array([[9.5, 9.5], [20.5, 20.5]])
    )
    shape.edge_width = 2
    npt.assert_array_equal(shape.bounding_box, np.array([[9, 9], [21, 21]]))
    shape.edge_width = 4
    npt.assert_array_equal(shape.bounding_box, np.array([[8, 8], [22, 22]]))


def test_rectangle_shift():
    shape = Rectangle(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-0.5, -0.5], [1.5, 1.5]])
    )

    shape.shift((1, 1))
    npt.assert_array_equal(
        shape.data, np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    )
    npt.assert_array_equal(
        shape.bounding_box, np.array([[0.5, 0.5], [2.5, 2.5]])
    )


def test_rectangle_rotate():
    shape = Rectangle(np.array([[1, 2], [-1, 2], [-1, -2], [1, -2]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-1.5, -2.5], [1.5, 2.5]])
    )
    shape.rotate(-90)
    npt.assert_array_almost_equal(
        shape.data, np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
    )
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-2.5, -1.5], [2.5, 1.5]])
    )


def test_nD_rectangle():
    """Test creating Shape with a single four corner planar 3D rectangle."""
    data = np.array(
        [[0, -10, -10], [0, -10, 20], [0, 20, 20], [0, 20, -10]],
        dtype=np.float32,
    )
    shape = Rectangle(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)


def test_polygon_data_triangle():
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ]
    )
    vertices, _triangles = PolygonData(vertices=data).triangulate()

    assert vertices.shape == (8, 2)


def test_polygon_data_triangle_module():
    pytest.importorskip('triangle')
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ]
    )
    vertices, _triangles = triangulate_face(data, triangulate_face_triangle)

    assert vertices.shape == (6, 2)


def test_polygon(triangulation_backend):
    """Test creating Shape with a random polygon."""
    # Test a single non convex six vertex polygon
    data = np.array(
        [
            [10.97627008, 14.30378733],
            [12.05526752, 10.89766366],
            [8.47309599, 12.91788226],
            [8.75174423, 17.83546002],
            [19.27325521, 7.66883038],
            [15.83450076, 10.5778984],
        ],
        dtype=np.float32,
    )
    shape = Polygon(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)
    # should get few triangles
    expected_face = (
        (8, 2)
        if triangulation_backend == TriangulationBackend.pure_python
        else (6, 2)
    )
    assert shape._edge_vertices.shape == (16, 2)
    assert shape._face_vertices.shape == expected_face


def test_polygon2():
    data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    shape = Polygon(data, interpolation_order=3)
    # should get many triangles

    expected_face = (249, 2)

    assert shape._edge_vertices.shape == (500, 2)
    assert shape._face_vertices.shape == expected_face


def test_polygon3():
    data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
    shape = Polygon(data, interpolation_order=3, ndisplay=3)
    # should get many vertices
    assert shape._edge_vertices.shape == (2500, 3)
    # faces are not made for non-coplanar 3d stuff
    assert shape._face_vertices.shape == (0, 3)


def test_nD_polygon():
    """Test creating Shape with a random nD polygon."""
    # Test a single six vertex planar 3D polygon
    np.random.seed(0)
    data = 20 * np.random.random((6, 3)).astype(np.float32)
    data[:, 0] = 0
    shape = Polygon(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (6, 3)


def test_path():
    """Test creating Shape with a random path."""
    # Test a single six vertex path
    np.random.seed(0)
    data = 20 * np.random.random((6, 2)).astype(np.float32)
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_path():
    """Test creating Shape with a random nD path."""
    # Test a single six vertex 3D path
    np.random.seed(0)
    data = 20 * np.random.random((6, 3)).astype(np.float32)
    shape = Path(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (6, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (6, 3)


def test_line():
    """Test creating 2D Line."""
    np.random.seed(0)
    data = np.array([[10, 10], [20, 20]], dtype=np.float32)
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_line():
    """Test creating Line in 3d"""
    data = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
    shape = Line(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (2, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (2, 3)


def test_ellipse1():
    """Test creating Ellipse by four corners."""
    data = np.array([(10, 10), (20, 10), (20, 20), (10, 20)], dtype=np.float32)
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_ellipse2():
    """Test creating Ellipse by upper left and lower right corners."""
    data = np.array([[10, 10], [20, 20]], dtype=np.float32)
    shape = Ellipse(data)
    assert len(shape.data) == 4
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 0)


def test_nD_ellipse():
    """Test creating Shape with a random nD ellipse."""
    # Test a single four corner planar 3D ellipse
    data = np.array(
        [(0, -10, -10), (0, -10, 20), (0, 20, 20), (0, 20, -10)],
        dtype=np.float32,
    )
    shape = Ellipse(data)
    np.testing.assert_array_equal(shape.data, data)
    assert shape.data_displayed.shape == (4, 2)
    assert shape.slice_key.shape == (2, 1)

    shape.ndisplay = 3
    assert shape.data_displayed.shape == (4, 3)


def test_ellipse_shift():
    shape = Ellipse(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-0.5, -0.5], [1.5, 1.5]])
    )

    shape.shift((1, 1))
    npt.assert_array_equal(
        shape.data, np.array([[1, 1], [2, 1], [2, 2], [1, 2]])
    )
    npt.assert_array_equal(
        shape.bounding_box, np.array([[0.5, 0.5], [2.5, 2.5]])
    )


def test_ellipse_rotate():
    shape = Ellipse(np.array([[1, 2], [-1, 2], [-1, -2], [1, -2]]))
    npt.assert_array_equal(
        shape.bounding_box, np.array([[-1.5, -2.5], [1.5, 2.5]])
    )
    shape.rotate(-90)
    npt.assert_array_almost_equal(
        shape.data, np.array([[-2, 1], [-2, -1], [2, -1], [2, 1]])
    )
    npt.assert_array_almost_equal(
        shape.bounding_box, np.array([[-2.5, -1.5], [2.5, 1.5]])
    )


# polygon with a hole: the edge connecting outer and inner ring is
# traversed twice and is not part of the polygon edge
POLYGON_WITH_HOLE = (
    np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0],
            [2, 5],
            [5, 8],
            [8, 5],
            [5, 2],
            [2, 5],
        ]
    )
    * 1.5
)


def _assert_same_mesh_region(shape, expected):
    """Check that two meshes cover the same region with the same vertices.

    Triangles are not compared directly, as they are not unique: some
    backends (bermuda) do not return them in a deterministic order, and
    a reflection may change which diagonal splits an edge segment.
    """

    def vertices_and_area(vertices, triangles):
        points = vertices[triangles].astype(np.float64)
        u = points[:, 1] - points[:, 0]
        v = points[:, 2] - points[:, 0]
        area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / 2
        # pure python backend may return vertices not used by any triangle
        used = vertices[np.unique(triangles)].astype(np.float64)
        return np.unique(used.round(3), axis=0), area

    assert shape.vertices_count == expected.vertices_count
    assert shape.triangles_count == expected.triangles_count
    for get_mesh in (
        lambda s: (s._face_vertices, s._face_triangles),
        lambda s: (
            s._edge_vertices + s.edge_width * s._edge_offsets,
            s._edge_triangles,
        ),
    ):
        vertices, area = vertices_and_area(*get_mesh(shape))
        expected_vertices, expected_area = vertices_and_area(
            *get_mesh(expected)
        )
        npt.assert_allclose(vertices, expected_vertices, atol=1e-3)
        npt.assert_allclose(area, expected_area, rtol=1e-5)


@pytest.mark.parametrize(
    ('method', 'arg'), [('rotate', 15), ('flip', 0), ('flip', 1)]
)
def test_polygon_orthogonal_transform_moves_mesh(method, arg):
    """Rotate and flip move the existing meshes instead of recomputing them."""
    shape = Polygon(POLYGON_WITH_HOLE)
    edge_triangles = shape._edge_triangles.copy()
    face_triangles = shape._face_triangles.copy()

    getattr(shape, method)(arg)

    npt.assert_array_equal(shape._edge_triangles, edge_triangles)
    npt.assert_array_equal(shape._face_triangles, face_triangles)
    _assert_same_mesh_region(shape, Polygon(shape.data))


def test_flip_with_center():
    data = np.array([[0, 0], [4, 0], [4, 2]])
    shape = Polygon(data)
    shape.flip(0, center=[1, 1])
    npt.assert_allclose(shape.data, [[0, 2], [4, 2], [4, 0]])


SQUARE = [[10, 10], [10, 50], [50, 50], [50, 10]]


@pytest.mark.parametrize('z', [5, 5.6])
@pytest.mark.parametrize(
    ('shape_class', 'data'),
    [
        (Polygon, SQUARE),
        (Rectangle, SQUARE),
        (Ellipse, SQUARE),
        (Path, [[10, 10], [30, 50], [50, 20]]),
        (Line, [[10, 10], [50, 50]]),
    ],
)
def test_to_mask_independent_of_ndisplay(shape_class, data, z):
    shape = shape_class(np.array([[z, *p] for p in data], dtype=float))
    mask_2d = shape.to_mask((10, 60, 60))
    assert mask_2d.any()

    shape.ndisplay = 3
    npt.assert_array_equal(shape.to_mask((10, 60, 60)), mask_2d)


@pytest.mark.parametrize('ndisplay', [2, 3])
@pytest.mark.parametrize(
    ('dims_order', 'vertices', 'expected_slice'),
    [
        ([1, 0, 2], [[p[0], 5, p[1]] for p in SQUARE], (slice(None), 5)),
        (
            [0, 2, 1, 3],
            [[1, p[0], 7, p[1]] for p in SQUARE],
            (1, slice(None), 7),
        ),
    ],
)
def test_to_mask_with_reordered_dims(
    dims_order, vertices, expected_slice, ndisplay
):
    shape = Polygon(np.array(vertices, dtype=float), dims_order=dims_order)
    shape.ndisplay = ndisplay
    mask_shape = (60, 10, 60) if len(dims_order) == 3 else (3, 60, 10, 60)
    mask = shape.to_mask(mask_shape)

    expected = np.zeros(mask_shape, dtype=bool)
    expected[expected_slice] = Polygon(np.array(SQUARE, dtype=float)).to_mask(
        (60, 60)
    )
    npt.assert_array_equal(mask, expected)
