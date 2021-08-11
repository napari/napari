import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.plane_manager import PlaneList, PlaneManager


def test_plane_manager_instantiation():
    plane = PlaneManager(position=(32, 32, 32), normal_vector=(1, 0, 0))
    assert isinstance(plane, PlaneManager)


def test_plane_manager_vector_normalisation():
    plane = PlaneManager(position=(0, 0, 0), normal_vector=(5, 0, 0))
    assert np.allclose(plane.normal, (1, 0, 0))


def test_plane_manager_vector_setter():
    plane = PlaneManager(position=(0, 0, 0), normal_vector=(1, 0, 0))
    plane.normal = (1, 0, 0)


def test_plane_manager_from_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    plane = PlaneManager.from_points(*points)
    assert isinstance(plane, PlaneManager)
    assert plane.normal == (0, 0, 1)
    assert np.allclose(plane.position, np.mean(points, axis=0))


def test_update_plane_manager_from_dict():
    properties = {
        'position': (0, 0, 0),
        'normal': (1, 0, 0),
        'enabled': True,
    }
    plane = PlaneManager()
    plane.update(properties)
    for k, v in properties.items():
        assert getattr(plane, k) == v


def test_plane_manager_from_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    plane = PlaneManager.from_array(array)
    assert isinstance(plane, PlaneManager)
    assert plane.position == pos
    assert plane.normal == norm


def test_plane_manager_to_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    plane = PlaneManager(position=pos, normal=norm)
    assert np.allclose(plane.as_array(), array)


def test_plane_manager_3_tuple():
    """Test for failure to instantiate with non 3-sequences of numbers"""
    with pytest.raises(ValidationError):
        plane = PlaneManager(  # noqa: F841
            position=(32, 32, 32, 32),
            normal_vector=(1, 0, 0, 0),
        )


def test_thick_plane_manager_instantiation():
    plane = PlaneManager(
        position=(32, 32, 32),
        normal_vector=(1, 0, 0),
        thickness=10,
    )
    assert isinstance(plane, PlaneManager)


def test_plane_list_instantiation():
    plane_list = PlaneList()
    assert isinstance(plane_list, PlaneList)


def test_plane_list_from_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    stacked = np.stack([array, array])
    plane_list = PlaneList.from_array(stacked)
    assert isinstance(plane_list, PlaneList)
    assert plane_list[0].position == pos
    assert plane_list[1].position == pos
    assert plane_list[0].normal == norm
    assert plane_list[1].normal == norm


def test_plane_list_as_array():
    pos = (0, 0, 0)
    norm = (0, 0, 1)
    array = np.array([pos, norm])
    stacked = np.stack([array, array])
    plane_list = PlaneList.from_array(stacked)
    assert np.allclose(plane_list.as_array(), array)


def test_plane_list_from_bounding_box():
    center = (0, 0, 0)
    dims = (2, 2, 2)
    plane_list = PlaneList.from_bounding_box(center, dims)
    assert isinstance(plane_list, PlaneList)
    assert len(plane_list) == 6
    assert plane_list.as_array().sum() == 0  # everything is mirrored around 0
