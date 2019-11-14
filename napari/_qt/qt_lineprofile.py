from .qt_plot_widget import QtPlotWidget
from vispy import scene


class QtLineProfile(QtPlotWidget):
    def __init__(self, layer=None, viewer=None, vertical=False):
        super().__init__(viewer=viewer, vertical=vertical)
        self.line = None

    def set_data(self, data):
        if not self.line:
            self.line = scene.LinePlot(
                data, color=(0.53, 0.56, 0.57, 1.00), width=1, marker_size=0
            )
            self.plot.view.add(self.line)
            self.plot.view.camera.set_range(margin=0.005)
        else:
            self.line.set_data(data, marker_size=0)
            # autoscale the range
            y = (0, data.max())
            x = (0, len(data))
            self.plot.view.camera.set_range(x=x, y=y, margin=0.005)
