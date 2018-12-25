from vispy.util.event import EmitterGroup, Event

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
        self.slider_values = (0, 1)
        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   update_slider=Event)
        self._qt = QtControlBars(self)

    def clim_slider_changed(self, slidermin, slidermax):
        self.slider_values = (slidermin, slidermax)
        msg = None
        for layer in self.viewer.layers:
            if hasattr(layer, 'visual') and layer.selected:
                valmin, valmax = layer._clim_range
                cmin = valmin+self.slider_values[0]*(valmax-valmin)
                cmax = valmin+self.slider_values[1]*(valmax-valmin)
                layer.clim = [cmin, cmax]
                msg = '(%.3f, %.3f)' % (cmin, cmax)
        if msg is not None:
            self.viewer._status = msg
            self.viewer.emit_status()

    def clim_slider_update(self):
        for layer in self.viewer.layers[::-1]:
            if hasattr(layer, 'visual') and layer.selected:
                valmin, valmax = layer._clim_range
                cmin, cmax = layer.clim
                slidermin = (cmin - valmin)/(valmax - valmin)
                slidermax = (cmax - valmin)/(valmax - valmin)
                msg = '(%.3f, %.3f)' % (cmin, cmax)
                self.viewer._status = msg
                self.viewer.emit_status()
                self.slider_values = (slidermin, slidermax)
                self.events.update_slider(values=self.slider_values, enabled=True)
                break
        else:
            self.events.update_slider(enabled=False)
