from .qt import QtControlBars

class ControlBars:
    """Controls object.

    Parameters
    ----------
    viewer : Viewer
        Parent viewer.
    """
    def __init__(self, viewer):
        self.viewer = viewer

        self._qt = QtControlBars()
        self._qt.climSlider.rangeChanged.connect(self.climSliderChanged)
        self._qt.mouseMoveEvent = self._mouseMoveEvent

    def climSliderChanged(self):
        slidermin, slidermax = self._qt.climSlider.getValues()
        msg = None
        for layer in self.viewer.layers:
            if hasattr(layer, 'visual') and layer.selected:
                valmin, valmax = layer._clim_range
                cmin = valmin+slidermin*(valmax-valmin)
                cmax = valmin+slidermax*(valmax-valmin)
                layer.clim = [cmin, cmax]
                msg = '(%.3f, %.3f)' % (cmin, cmax)
        if msg is not None:
            self.viewer._status = msg
            self.viewer.emitStatus()

    def climSliderUpdate(self):
        for layer in self.viewer.layers[::-1]:
            if hasattr(layer, 'visual') and layer.selected:
                valmin, valmax = layer._clim_range
                cmin, cmax = layer.clim
                slidermin = (cmin - valmin)/(valmax - valmin)
                slidermax = (cmax - valmin)/(valmax - valmin)
                self._qt.climSlider.setValues((slidermin, slidermax))
                msg = '(%.3f, %.3f)' % (cmin, cmax)
                self.viewer._status = msg
                self.viewer.emitStatus()
                self._qt.climSlider.setEnabled(True)
                break
        else:
            self._qt.climSlider.setEnabled(False)

    def _mouseMoveEvent(self, event):
        for layer in self.viewer.layers[::-1]:
            if hasattr(layer, 'visual') and layer.selected:
                cmin, cmax = layer.clim
                msg = '(%.3f, %.3f)' % (cmin, cmax)
                self.viewer._status = msg
                self.viewer.emitStatus()
                break
