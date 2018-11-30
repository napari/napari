import numpy as np

from .qt import QtControls


class Controls:
    """Controls object.
    """
    def __init__(self, viewer, layer_list):
        self.viewer = viewer
        self.layer_list = layer_list

        clim_min, clim_max = self.calc_clim_minmax()
        self._qt = QtControls(clim_min, clim_max)
        self._qt.climSlider.rangeChanged.connect(self.climSliderChanged)

    def calc_clim_minmax(self):
        # Initialize
        clim_min, clim_max = 0, 255  # TODO: check if initial values make sense

        # Iterate over the layer_list and get min/max values
        for layer in self.layer_list._list:
            if layer.selected:
                image_data = layer.image
                clim_min = min(clim_min, np.min(image_data))
                clim_max = max(clim_max, np.max(image_data))
                print(np.min(image_data), np.max(image_data))

        return clim_min, clim_max

    def climSliderChanged(self):
        for layer in self.layer_list._list:
            if layer.selected and layer.visual is not None:
                layer.visual.clim = list(self.calc_clim_minmax())


