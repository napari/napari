from gui.components._dims.model import Mode
from napari_gui.components import Dims


def test_modes():

    dims = Dims(None)

    dims.set_mode(3, Mode.Display)

    print(dims.mode)

    assert dims.mode[0] == None
    assert dims.mode[1] == None
    assert dims.mode[2] == None
    assert dims.mode[3] == Mode.Display


def test_point():

    dims = Dims(None)

    dims.set_point(3, Mode.Display)

    print(dims.mode)

    assert dims.mode[0] == None
    assert dims.mode[1] == None
    assert dims.mode[2] == None
    assert dims.mode[3] == Mode.Display
