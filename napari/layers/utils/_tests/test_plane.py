import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.plane import ClippingPlaneList, Plane, SlicingPlane


def test_plane_instantiation():
    plane = Plane(position=(32, 32, 32), normal=(1, 0, 0), thickness=2)
    assert isinstance(plane, Plane)


def test_plane_vector_normalisation():
    plane = Plane(position=(0, 0, 0), normal=(5, 0, 0))
    assert np.allclose(plane.normal, (1, 0, 0))


def test_plane_vector_setter():
    plane = Plane(position=(0, 0, 0), normal=(1, 0, 0))
    plane.normal = (1, 0, 0)


def test_plane_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    plane = Plane.from_points(*points)
    assert isinstance(plane, Plane)
    assert plane.normal == (0, 0, 1)
    assert np.allclose(plane.position, np.mean(points, axis=0))


def test_update_slicing_plane_from_dict():
    properties = {
        'position': (0, 0, 0),
        'normal': (1, 0, 0),
    }
    plane = SlicingPlane()
    plane.update(properties)
    for k, v in properties.items():
        assert getattr(plane, k) == v


def test_plane_from_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    plane = SlicingPlane.from_array(array)
    assert isinstance(plane, SlicingPlane)
    assert plane.position == pos
    assert plane.normal == norm


def test_plane_to_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    plane = SlicingPlane(position=pos, normal=norm)
    assert np.allclose(plane.as_array(), array)


def test_plane_3_tuple():
    """Test for failure to instantiate with non 3-sequences of numbers"""
    with pytest.raises(ValidationError):
        plane = SlicingPlane(  # noqa: F841
            position=(32, 32, 32, 32),
            normal=(1, 0, 0, 0),
        )


def test_clipping_plane_list_instantiation():
    plane_list = ClippingPlaneList()
    assert isinstance(plane_list, ClippingPlaneList)


def test_clipping_plane_list_from_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    stacked = np.stack([array, array])
    plane_list = ClippingPlaneList.from_array(stacked)
    assert isinstance(plane_list, ClippingPlaneList)
    assert plane_list[0].position == pos
    assert plane_list[1].position == pos
    assert plane_list[0].normal == norm
    assert plane_list[1].normal == norm


def test_clipping_plane_list_as_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    stacked = np.stack([array, array])
    plane_list = ClippingPlaneList.from_array(stacked)
    assert np.allclose(plane_list.as_array(), array)


def test_clipping_plane_list_from_bounding_box():
    center = (0, 0, 0)
    dims = (2, 2, 2)
    plane_list = ClippingPlaneList.from_bounding_box(center, dims)
    assert isinstance(plane_list, ClippingPlaneList)
    assert len(plane_list) == 6
    assert plane_list.as_array().sum() == 0  # everything is mirrored around 0
