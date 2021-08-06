import numpy as np

from napari.layers.utils.plane import Plane3D


def test_Plane3D_instantiation():
    plane = Plane3D(
        position=(32, 32, 32), normal_vector=(1, 0, 0), thickness=10
    )
    assert isinstance(plane, Plane3D)


def test_Plane3D_vector_normalisation():
    plane = Plane3D(position=(0, 0, 0), normal_vector=(5, 0, 0), thickness=10)
    assert np.allclose(plane.normal_vector, (1, 0, 0))
