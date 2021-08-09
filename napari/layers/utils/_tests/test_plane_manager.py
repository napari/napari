import numpy as np
import pytest
from pydantic import ValidationError

from napari.layers.utils.plane_manager import PlaneManager


def test_plane_manager_instantiation():
    plane = PlaneManager(
        position=(32, 32, 32), normal_vector=(1, 0, 0), thickness=10
    )
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
        'thickness': 10,
        'enabled': True,
    }
    plane = PlaneManager()
    plane.update(properties)
    for k, v in properties.items():
        assert getattr(plane, k) == v


def test_plane_manager_3_tuple():
    """Test for failure to instantiate with non 3-sequences of numbers"""
    with pytest.raises(ValidationError):
        plane = PlaneManager(  # noqa: F841
            position=(32, 32, 32, 32), normal_vector=(1, 0, 0, 0), thickness=10
        )
