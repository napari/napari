from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtGui import QGuiApplication
from .._vispy.vispy_histogram import HistogramScene
from vispy import scene
import numpy as np


class HistogramWidget(QWidget):

    clims_updated = Signal(tuple)
    gamma_updated = Signal(float)

    def __init__(
        self, layer=None, clims=None, gamma=1, clim_handle_color=(1, 0, 0, 1)
    ):
        super().__init__()
        self._clims = clims
        self._gamma = gamma
        data = None
        if layer is not None:
            data = layer.data
            if not clims:
                self._clims = layer.contrast_limits
        self.canvas = HistogramScene(data, keys=None, vsync=True)
        self.plot = self.canvas.plot
        self.camera = self.canvas.plot.view.camera

        self.canvas.events.ignore_callback_errors = False
        # self.canvas.events.draw.connect(self.dims.enable_play)
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.canvas.native.resize(800, 200)

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        # self.canvas.connect(self.on_key_press)
        # self.canvas.connect(self.on_key_release)
        # self.canvas.connect(self.on_draw)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.addWidget(self.canvas.native)
        self.canvas.native.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        self.clim_handle_color = clim_handle_color
        self.clim_handles = []
        _, y1 = self.range
        for clim in self.clims:
            coord = np.array([(clim, 0), (clim, y1)])
            line = scene.Line(coord, self.clim_handle_color)
            self.clim_handles.append(line)
            self.plot.view.add(line)

        midpoint = np.array([(np.mean(self.clims), y1 * 2 ** -self.gamma)])
        self.gamma_handle = scene.Markers(pos=midpoint, size=6, edge_width=0)
        self.plot.view.add(self.gamma_handle)

        self.lut_line = None
        self.update_lut_line()
        self._clim_handle_grabbed = 0
        self._gamma_handle_grabbed = 0

        self.resize(self.layout.sizeHint())
        self.camera.set_range(margin=0.005)
        if layer is not None:
            self.link_layer(layer)

    def update_lut_line(self):
        npoints = 255
        y1 = self.range[1]
        X = np.linspace(self.clims[0], self.clims[1], npoints)
        Y = np.linspace(0, 1, npoints) ** self.gamma * y1
        if self.lut_line:
            self.lut_line.set_data(
                (X, Y), color=self.clim_handle_color, marker_size=0
            )
        else:
            self.lut_line = self.plot.plot(
                (X, Y), color=self.clim_handle_color
            )

        midpoint = np.array([(np.mean(self.clims), y1 * 2 ** -self.gamma)])
        self.gamma_handle.set_data(pos=midpoint, size=6, edge_width=0)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not np.isscalar(value):
            raise ValueError('gamma value must be a scalar')
        self._gamma = float(value)
        self.update_lut_line()
        self.gamma_updated.emit(value)

    @property
    def clims(self):
        return self._clims

    @clims.setter
    def clims(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
            raise ValueError('clims value must be a 2-item array')

        for i in range(2):
            if self._clims[i] != value[i]:
                coord = np.array([(value[i], 0), (value[i], self.range[1])])
                self.clim_handles[i].set_data(coord)
                # self.clim_handles[i].update()
                self.update_lut_line()
        self._clims = value
        self.clims_updated.emit(value)

    @property
    def domain(self):
        return self.plot.xaxis.axis.domain

    @property
    def range(self):
        return self.plot.yaxis.axis.domain

    def _get_mouse_coords(self, pos):
        tr = self.plot.node_transform(self.plot.view.scene)
        x, y, _, _ = tr.map(pos)
        return x, y

    def on_mouse_press(self, event):
        if event.pos is None:
            return
        # determine if a clim_handle was clicked
        x, y = self._get_mouse_coords(event.pos)
        self._clim_handle_grabbed = self._pos_is_clim(x)
        self._gamma_handle_grabbed = self._pos_is_gamma(x, y)
        if self._clim_handle_grabbed or self._gamma_handle_grabbed:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.camera._viewbox_unset(self.camera.viewbox)

    def on_mouse_release(self, event):
        self._clim_handle_grabbed = 0
        self._gamma_handle_grabbed = 0
        if not self.camera.viewbox:
            self.camera._viewbox_set(self.plot.view)

    def _pos_is_clim(self, x):
        """Returns 1 if x is near clims[0], 2 if near clims[1], else 0"""
        # FIXME: strategy doesn't work when zoomed in
        if abs(self.clims[0] - x) < 2:
            return 1
        if abs(self.clims[1] - x) < 2:
            return 2
        return 0

    def _pos_is_gamma(self, x, y):
        """Returns True if value is near the gamma handle"""
        gx, gy, _ = self.gamma_handle._data[0][0]
        if abs(gx - x) < 3 and abs(gy - y) < 100:
            return True
        return False

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return

        x, y = self._get_mouse_coords(event.pos)

        if self._clim_handle_grabbed:
            newlims = list(self.clims)
            newlims[self._clim_handle_grabbed - 1] = x
            self.clims = newlims
            return

        if self._gamma_handle_grabbed:
            y0, y1 = self.range
            if y < np.maximum(y0, 0) or y > y1:
                return
            self.gamma = -np.log2(y / y1)
            return

        QGuiApplication.restoreOverrideCursor()

        if self._pos_is_clim(x):
            QGuiApplication.setOverrideCursor(Qt.CursorShape.SplitHCursor)
        elif self._pos_is_gamma(x, y):
            QGuiApplication.setOverrideCursor(Qt.CursorShape.SplitVCursor)
        else:
            x1, x2 = self.domain
            y1, y2 = self.range
            if (x1 < x <= x2) and (y1 <= y <= y2):
                QGuiApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)

    def link_layer(self, layer):
        def set_self_gamma(x):
            self.gamma = layer.gamma

        def set_layer_gamma(x):
            with layer.events.gamma.blocker(set_self_gamma):
                layer.gamma = x

        def set_self_clims(x):
            self.clims = layer.contrast_limits

        def set_layer_clims(x):
            with layer.events.contrast_limits.blocker(set_self_clims):
                layer.contrast_limits = x

        self.gamma_updated.connect(set_layer_gamma)
        layer.events.gamma.connect(set_self_gamma)
        self.clims_updated.connect(set_layer_clims)
        layer.events.contrast_limits.connect(set_self_clims)


if __name__ == '__main__':
    from qtpy.QtWidgets import QApplication
    import sys
    from skimage.io import imread

    data = imread('/Users/talley/Desktop/test.tif')

    app = QApplication([])

    widg = HistogramWidget(data[0])

    widg.show()
    sys.exit(app.exec_())
