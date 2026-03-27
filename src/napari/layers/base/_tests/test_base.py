from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pint
import pytest

from napari.layers.base._test_util_sample_layer import SampleLayer

REG = pint.get_application_registry()


@pytest.mark.parametrize(
    ('units', 'expected'),
    [
        (('nm', 'nm'), (REG.nm, REG.nm)),
        ((REG.nm, REG.nm), (REG.nm, REG.nm)),
        ('nm', (REG.nm, REG.nm)),
        (None, (REG.pixel, REG.pixel)),
        ((None, None), (REG.pixel, REG.pixel)),
        ((None, 'nm'), (REG.pixel, REG.nm)),
    ],
)
def test_assign_units(units, expected):
    layer = SampleLayer(np.empty((10, 10)), units=['mm', 'mm'])
    mock = Mock()
    layer.events.units.connect(mock)
    layer.units = units
    assert layer.units == expected
    mock.assert_called_once()


def test_no_emmit_on_identical_units():
    layer = SampleLayer(np.empty((10, 10)))
    mock = Mock()
    layer.events.units.connect(mock)
    layer.units = ('nm', 'nm')
    assert layer.units == (REG.nm, REG.nm)
    mock.assert_called_once()

    layer.units = ('nm', REG.nm)
    mock.assert_called_once()

    layer.units = (REG.nm, REG.nm)
    mock.assert_called_once()


def test_exception_on_invalid_units():
    layer = SampleLayer(np.empty((10, 10)))
    with pytest.raises(ValueError, match='Could not find unit'):
        layer.units = ('ugh', 'ugh')

    with pytest.raises(ValueError, match='Could not find unit'):
        layer.units = (1, 1)

    with pytest.raises(ValueError, match='Could not find unit'):
        layer.units = 1


def test_units_constructor():
    layer = SampleLayer(np.empty((10, 10)), units=('nm', 'nm'))
    assert layer.units == (REG.nm, REG.nm)

    layer = SampleLayer(np.empty((10, 10)), units=(REG.mm, REG.mm))
    assert layer.units == (REG.mm, REG.mm)

    layer = SampleLayer(np.empty((10, 10)), units=('mm', 'mm'))
    assert layer.units == (REG.mm, REG.mm)

    layer = SampleLayer(np.empty((10, 10)), units=None)
    assert layer.units == (REG.pixel, REG.pixel)


def test_assign_units_error():
    layer = SampleLayer(np.empty((10, 10)))
    with pytest.raises(ValueError, match='must have length ndim'):
        layer.units = ('m', 'm', 'm')

    with pytest.raises(ValueError, match='Could not find unit'):
        layer.units = ('ugh', 'ugh')

    with pytest.raises(ValueError, match='Could not find unit'):
        SampleLayer(np.empty((10, 10)), units=('ugh', 'ugh'))

    with pytest.raises(ValueError, match='must have length ndim'):
        SampleLayer(np.empty((10, 10)), units=('m', 'm', 'm'))


def test_axis_labels_assign():
    layer = SampleLayer(np.empty((10, 10)))
    mock = Mock()
    layer.events.axis_labels.connect(mock)
    assert layer.axis_labels == ('-2', '-1')

    layer.axis_labels = ('x', 'y')
    mock.assert_called_once()
    mock.reset_mock()

    assert layer.axis_labels == ('x', 'y')

    layer.axis_labels = ('x', 'y')
    mock.assert_not_called()

    layer.axis_labels = None
    mock.assert_called_once()
    assert layer.axis_labels == ('-2', '-1')


def test_axis_labels_constructor():
    layer = SampleLayer(np.empty((10, 10)), axis_labels=('x', 'y'))
    assert layer.axis_labels == ('x', 'y')

    layer = SampleLayer(np.empty((10, 10)), axis_labels=None)
    assert layer.axis_labels == ('-2', '-1')


def test_axis_labels_error():
    layer = SampleLayer(np.empty((10, 10)))
    with pytest.raises(ValueError, match='must have length ndim'):
        layer.axis_labels = ('x', 'y', 'z')

    with pytest.raises(ValueError, match='must have length ndim'):
        SampleLayer(np.empty((10, 10)), axis_labels=('x', 'y', 'z'))


def test_non_visible_mode():
    layer = SampleLayer(np.empty((10, 10)))
    layer.mode = 'transform'

    # change layer visibility and check the layer mode gets updated
    layer.visible = False
    assert layer.mode == 'pan_zoom'
    layer.visible = True
    assert layer.mode == 'transform'


def test_world_to_displayed_data_normal_3D():
    layer = SampleLayer(np.empty((10, 10, 10)))
    layer.scale = (1, 3, 2)

    normal_vector = [0, 1, 1]

    expected_transformed_vector = [0, 3 * (13**0.5) / 13, 2 * (13**0.5) / 13]

    transformed_vector = layer._world_to_displayed_data_normal(
        normal_vector, dims_displayed=[0, 1, 2]
    )

    assert np.allclose(transformed_vector, expected_transformed_vector)


def test_world_to_displayed_data_normal_4D():
    layer = SampleLayer(np.empty((10, 10, 10, 10)))
    layer.scale = (1, 3, 2, 1)

    normal_vector = [0, 1, 1]

    expected_transformed_vector = [0, 3 * (13**0.5) / 13, 2 * (13**0.5) / 13]

    transformed_vector = layer._world_to_displayed_data_normal(
        normal_vector, dims_displayed=[0, 1, 2]
    )

    assert np.allclose(transformed_vector, expected_transformed_vector)


def test_invalidate_extent_scale():
    """Test that the extent is invalidated when the data changes."""
    layer = SampleLayer(np.empty((10, 10)))
    npt.assert_array_equal(layer.extent.step, (1, 1))
    with layer._block_refresh():
        layer.scale = (2, 2)
    npt.assert_array_equal(layer.extent.step, (2, 2))


def test_invalidate_extent_units():
    """Test that the extent is invalidated when the data changes."""
    # commented lines required 7889 and should be uncomment later
    layer = SampleLayer(np.empty((10, 10)))
    # px = pint.get_application_registry().pixel
    # mm = pint.get_application_registry().mm
    # npt.assert_array_equal(layer.extent.units, (px, px))
    with layer._block_refresh():
        layer.units = ('mm', 'mm')
    # npt.assert_array_equal(layer.extent.units, (mm, mm))


def test_invalidate_extent_translate():
    """Test that the extent is invalidated when the data changes."""
    layer = SampleLayer(np.empty((10, 10)))
    npt.assert_array_equal(layer.extent.world[0], (0, 0))
    with layer._block_refresh():
        layer.translate = (1, 1)
    npt.assert_array_equal(layer.extent.world[0], (1, 1))


def test_invalidate_extent_rotate():
    """Test that the extent is invalidated when the data changes."""
    layer = SampleLayer(np.empty((10, 20)))
    npt.assert_array_equal(layer.extent.world, [[0, 0], [9, 19]])
    with layer._block_refresh():
        layer.rotate = 90
    npt.assert_almost_equal(layer.extent.world, [[-19, 0], [0, 9]])


def test_invalidate_extent_affine():
    """Test that the extent is invalidated when the data changes."""
    layer = SampleLayer(np.empty((10, 20)))
    npt.assert_array_equal(layer.extent.world, [[0, 0], [9, 19]])
    with layer._block_refresh():
        layer.affine = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    npt.assert_array_equal(layer.extent.world, [[0, 0], [18, 38]])


def test_invalidate_extent_shear():
    """Test that the extent is invalidated when the data changes."""
    layer = SampleLayer(np.empty((10, 20)))
    npt.assert_array_equal(layer.extent.world, [[0, 0], [9, 19]])
    with layer._block_refresh():
        layer.shear = [1]
    npt.assert_array_equal(layer.extent.world, [[0, 0], [28, 19]])


def test_get_ray_intersections_anisotropic():
    """Regression test for #8285.

    With highly anisotropic data (small z, large y/x) the old
    face-detection approach failed to identify both bounding-box faces,
    causing a TypeError.  The slab-based intersection now handles
    arbitrary aspect ratios and returns valid intersection points.
    """
    data = np.zeros((5, 5000, 5000))
    layer = SampleLayer(data)

    # Position and direction from the original issue traceback.
    position = np.array([5.10589, 3717.37829, 3671.51104])
    view_direction = np.array([-1.97862e-04, -6.36407e-01, 7.71354e-01])
    dims_displayed = [0, 1, 2]

    # The ray does intersect the bounding box, so we expect valid points
    start_point, end_point = layer.get_ray_intersections(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=False,
    )
    assert start_point is not None
    assert end_point is not None
    # Both points should lie on the bounding box faces
    bb_min = np.array([0, 0, 0])
    bb_max = np.array([6, 5001, 5001])  # extent is shape + 1
    for pt in (start_point, end_point):
        assert np.all(pt >= bb_min - 1e-6) and np.all(pt <= bb_max + 1e-6)


def test_get_ray_intersections_miss():
    """Ray that misses the bounding box entirely returns (None, None)."""
    data = np.zeros((5, 5, 5))
    layer = SampleLayer(data)

    # Position far outside, direction pointing away
    position = np.array([100.0, 100.0, 100.0])
    view_direction = np.array([1.0, 0.0, 0.0])
    dims_displayed = [0, 1, 2]

    start_point, end_point = layer.get_ray_intersections(
        position,
        view_direction=view_direction,
        dims_displayed=dims_displayed,
        world=False,
    )
    assert start_point is None
    assert end_point is None
