from napari.components import Dims
from napari.components._dims._constants import DimsMode


def test_point():
    """
    Tests point setting
    """
    dims = Dims(4)
    dims.set_point(3, 2.5)

    assert dims.ndim == 4
    assert dims.point[0] == 0.0
    assert dims.point[1] == 0.0
    assert dims.point[2] == 0.0
    assert dims.point[3] == 2.5


def test_display():
    """
    Tests display setting
    """
    dims = Dims(4)
    dims.set_display(0, False)
    dims.set_display(1, False)
    dims.set_display(2, True)
    dims.set_display(3, True)

    assert dims.ndim == 4
    assert not dims.display[0]
    assert not dims.display[1]
    assert dims.display[2]
    assert dims.display[3]
    assert (dims.displayed == [2, 3]).all()


def test_add_remove_dims():
    """
    Tests adding and removing dimensions
    """
    dims = Dims(2)
    assert dims.ndim == 2

    dims.ndim = 10
    assert dims.ndim == 10

    dims.ndim = 5
    assert dims.ndim == 5
