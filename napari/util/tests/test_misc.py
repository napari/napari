from ..misc import is_multichannel


def test_is_multichannel():
    meta = {}

    meta['itype'] = 'rgb'
    assert is_multichannel(meta)

    meta['itype'] = 'rgba'
    assert is_multichannel(meta)

    meta['itype'] = 'multi'
    assert is_multichannel(meta)

    meta['itype'] = 'multichannel'
    assert is_multichannel(meta)

    meta['itype'] = 'notintupleforsure'
    assert not is_multichannel(meta)
