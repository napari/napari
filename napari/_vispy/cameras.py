from vispy.scene.cameras import PanZoomCamera
import numpy as np


class PanZoom1DCamera(PanZoomCamera):
    def __init__(self, axis=1, *args, **kwargs):
        """A camera that can only Pan/Zoom along one axis.

        Useful in a PlotWidget.
        
        Parameters
        ----------
        axis : int, optional
            The axis to constrain. (This axis will NOT pan or zoom).
                0 => lock x axis
                1 => lock y axis
                by default 1
        """
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
