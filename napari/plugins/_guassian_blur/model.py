from skimage.filters import gaussian

from .._base_plugin import Plugin
from .view import QtGaussianBlur


class GaussianBlur(Plugin):
    """GaussianBlur plugin class.
    """
    def __init__(self):
        super().__init__()

        self._blur = 1
        self._qt = QtGaussianBlur(self)

    @property
    def blur(self):
        """float: Gaussian blur to apply to target image
        """
        return self._blur

    @blur.setter
    def blur(self, blur):
        if blur == self.blur:
            return
        self._blur = blur
        self.run()

    def run(self):
        self.layers[1].image = 255*gaussian(self.layers[0].image, self.blur)
