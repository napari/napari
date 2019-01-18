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
        self.values = (0, 1)
        self.events = EmitterGroup(source=self,
                                   auto_connect=True,
                                   update_slider=Event)
        self._qt = QtControlBars(self)

    def clim_slider_update(self):
        # iteration goes backwards to find top most visible layer
        for layer in self.viewer.layers[::-1]:
            if hasattr(layer, 'visual') and layer.selected:
                valmin, valmax = layer._clim_range
                cmin, cmax = layer.clim
                slidermin = (cmin - valmin)/(valmax - valmin)
                slidermax = (cmax - valmin)/(valmax - valmin)
                msg = f'({cmin:0.3}, {cmax:0.3})'
                self.viewer.status = msg
                self.values = (slidermin, slidermax)
                self.events.update_slider(values=self.values,
                                          enabled=True)
                break
        else:
            self.events.update_slider(enabled=False)
            self.viewer.status = 'Ready'
