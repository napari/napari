import pytest

import misc


class MiscTest(object):

    def test_is_multichannel(self):
        meta = {}

        meta['itype'] = 'rgb'
        assert misc.is_multichannel(meta) == True

        meta['itype'] = 'rgba'
        assert misc.is_multichannel(meta) == True

        meta['itype'] = 'multi'
        assert misc.is_multichannel(meta) == True

        meta['itype'] = 'multichannel'
        assert misc.is_multichannel(meta) == True

        meta['itype'] = 'notintupleforsure'
        assert misc.is_multichannel(meta) == False
