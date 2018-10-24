import unittest

from ...gui.util import misc


class MiscTest(unittest.TestCase):

    def test_is_multichannel(self):
        meta = {}

        meta['itype'] = 'rgb'
        self.assertTrue(misc.is_multichannel(meta))

        meta['itype'] = 'rgba'
        self.assertTrue(misc.is_multichannel(meta))

        meta['itype'] = 'multi'
        self.assertTrue(misc.is_multichannel(meta))
        
        meta['itype'] = 'multichannel'
        self.assertTrue(misc.is_multichannel(meta))
        
        meta['itype'] = 'notintupleforsure'
        self.assertFalse(misc.is_multichannel(meta)) 

