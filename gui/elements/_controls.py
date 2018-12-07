import numpy as np

from .qt import QtControls


class Controls:
    """Controls object.
    """
    def __init__(self, viewer):
        self.viewer = viewer

        self._qt = QtControls()
        self._qt.climSlider.rangeChanged.connect(self.climSliderChanged)

    def calc_clim_minmax(self):
        # Initialize
        clim_min, clim_max = float("inf"), -float("inf")

        # Iterate over the layer_list and get min/max values
        for layer in self.viewer.layers:
            image_data = layer.image
            clim_min = min(clim_min, np.min(image_data))
            clim_max = max(clim_max, np.max(image_data))
            # print("this are layers", np.min(image_data), np.max(image_data))

        if clim_min == float("inf"):
            clim_min = 0
        if clim_max == -float("inf"):
            clim_max = 1

        return clim_min, clim_max

    def climSliderChanged(self):
        valmin, valmax = self.calc_clim_minmax()
        # print("val", valmin, valmax)
        slidermin, slidermax = self._qt.climSlider.getValues()
        # print("slider", slidermin, slidermax)

        dismin = valmin*0.8+slidermin*(1.2*valmax-0.8*valmin)
        dismax = valmin*0.8+slidermax*(1.2*valmax-0.8*valmin)
        # print("display",dismin,dismax)

        for layer in self.viewer.layers:
            if layer.visual is not None:
                layer.visual.clim = [dismin, dismax]
