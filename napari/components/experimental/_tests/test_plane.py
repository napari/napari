from .._plane import Plane, PlaneList, Slice


def test_plane_instantiation():
    plane = Plane(position=(64, 64, 64), normal=(1, 1, 1))
    assert isinstance(plane, Plane)


def test_planelist_instantiation(plane):
    planes = PlaneList([plane for _ in range(5)])
    assert isinstance(planes, PlaneList)


def test_slice_instantiation(plane):
    slice = Slice(plane=plane, thickness=10)
    assert isinstance(slice, Slice)
