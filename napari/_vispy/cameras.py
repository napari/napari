from vispy.scene.cameras import PanZoomCamera
import numpy as np


class PanZoom1DCamera(PanZoomCamera):
    def __init__(self, axis=1, *args, **kwargs):
        self.axis = axis
        super().__init__(*args, **kwargs)

    def zoom(self, factor, center=None):
        if np.isscalar(factor):
            factor = [factor, factor]
        factor[self.axis] = 1
        return super().zoom(factor, center=center)

    def pan(self, pan):
        pan[self.axis] = 0
        self.rect = self.rect + pan
