from napari.components import Dims
from napari.components._dims.model import Mode


def test_range():
    """
    Tests mode initialisation:
    """
    dims = Dims(2)


    dims.set_range(3, (0,1000,0.1))

    print(dims.range)

    assert dims.nb_dimensions == 4

    assert dims.range[0] == (None,None,None)
    assert dims.range[1] == (None,None,None)
    assert dims.range[2] == (None,None,None)
    assert dims.range[3] == (0,1000,0.1)

def test_mode():
    """
    Tests mode initialisation:
    """
    dims = Dims(2)

    dims.set_mode(3, Mode.Interval)

    print(dims.mode)

    assert dims.nb_dimensions == 4

    assert dims.mode[0] is None
    assert dims.mode[1] is None
    assert dims.mode[2] is None
    assert dims.mode[3] == Mode.Interval


def test_point():
    """
    Tests point initialisation
    """
    dims = Dims(2)

    dims.set_point(3, 2.5)

    print(dims.point)

    assert dims.nb_dimensions == 4

    assert dims.point[0] == 0.0
    assert dims.point[1] == 0.0
    assert dims.point[2] == 0.0
    assert dims.point[3] == 2.5


def test_interval():
    """
    Tests interval initialisation
    """
    dims = Dims(2)

    dims.set_interval(3, (0, 10))

    print(dims.interval)

    assert dims.nb_dimensions == 4

    assert dims.interval[0] == None
    assert dims.interval[1] == None
    assert dims.interval[2] == None
    assert dims.interval[3] == (0, 10)


def test_display():
    """
    Tests display initialisation
    """
    dims = Dims(2)

    dims.set_display(0, False)
    dims.set_display(1, False)
    dims.set_display(2, True)
    dims.set_display(3, True)

    print(dims.interval)

    assert dims.nb_dimensions == 4

    assert not dims.display[0]
    assert not dims.display[1]
    assert dims.display[2]
    assert dims.display[3]


def test_slice_and_project():
    dims = Dims(2)

    dims.set_point(3, 2.5)

    dims.set_mode(0, Mode.Point)
    dims.set_mode(1, Mode.Point)
    dims.set_mode(2, Mode.Interval)
    dims.set_mode(3, Mode.Interval)

    print(dims.slice_and_project)

    (sliceit, projectit) = dims.slice_and_project

    assert sliceit[0] == slice(None, 0, None)
    assert sliceit[1] == slice(None, 0, None)
    assert sliceit[2] == slice(None, None, None)
    assert sliceit[3] == slice(None, None, None)

    assert projectit[0] == False
    assert projectit[1] == False
    assert projectit[2] == True
    assert projectit[3] == True
