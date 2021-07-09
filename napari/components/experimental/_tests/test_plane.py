from .._plane import Plane, PlaneList, ThickPlane


def test_plane_instantiation():
    plane = Plane(position=(64, 64, 64), normal=(1, 1, 1))
    assert isinstance(plane, Plane)


def test_planelist_instantiation(plane):
    planes = PlaneList([plane for _ in range(5)])
    assert isinstance(planes, PlaneList)


def test_thick_plane_instantiation(plane):
    slice = ThickPlane(plane=plane, thickness=10)
    assert isinstance(slice, ThickPlane)
