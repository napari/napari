from ..misc import is_multichannel


def test_is_multichannel():
    shape = (10, 15)
    assert not is_multichannel(shape)

    shape = (10, 15, 6)
    assert not is_multichannel(shape)

    shape = (10, 15, 3)
    assert is_multichannel(shape)

    shape = (10, 15, 4)
    assert is_multichannel(shape)
