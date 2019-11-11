from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtGui import QGuiApplication
from .._vispy.vispy_plot import NapariPlotWidget
from .._vispy.vispy_histogram import VispyHistogramLayer
from vispy import scene
import numpy as np


class QtHistogramWidget(QWidget):

    clims_updated = Signal(tuple)
    gamma_updated = Signal(float)

    def __init__(
        self,
        layer=None,
        viewer=None,
        clims=(0, 255),
        gamma=1,
        clim_handle_color=(0.26, 0.28, 0.31, 1),
        orientation='v',
    ):
        super().__init__()

        self.hist_layer = VispyHistogramLayer(
            link='view', orientation=orientation
        )
        self._viewer = viewer
        self.orientation = orientation

        self.canvas = scene.SceneCanvas(bgcolor='k', keys=None, vsync=True)
        self.canvas.events.ignore_callback_errors = False
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.canvas.native.resize(800, 200)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_resize)
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

        self.plot = self.canvas.central_widget.add_widget(
            NapariPlotWidget(
                fg_color=(1, 1, 1, 0.3),
                show_yaxis=False,
                lock_axis=(1 if orientation == 'h' else 0),
            )
        )
        self.plot._configure_2d()
        self.camera = self.plot.view.camera
        self.camera.set_range(margin=0.005)

        self._clims = clims
        self._gamma = gamma
        self._layer = None

        self.clim_handle_color = clim_handle_color
        self.clim_handles = []
        _, y1 = self.range
        for clim in self.clims:
            line = scene.InfiniteLine(clim, self.clim_handle_color)
            self.clim_handles.append(line)
            self.plot.view.add(line)

        midpoint = np.array([(np.mean(self.clims), y1 * 2 ** -self.gamma)])
        self.gamma_handle = scene.Markers(pos=midpoint, size=6, edge_width=0)
        self.plot.view.add(self.gamma_handle)

        self.lut_line = None
        self._clim_handle_grabbed = 0
        self._gamma_handle_grabbed = 0

        if self._viewer:
            # viewer takes precedence
            self.layer = self._viewer.active_layer

            def change_layer(event):
                self.layer = event.item

            self._viewer.events.active_layer.connect(change_layer)
        else:
            self.layer = layer
        self.plot.view.add(self.hist_layer.node)
        self.resize(self.layout.sizeHint())
        self.autoscale()
        self.update_lut_line()

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, newlayer):
        if newlayer == self._layer:
            return
        if self._layer is not None:
            self.unlink_layer()
        self._layer = newlayer
        if newlayer is not None:
            self._link_layer(newlayer)
            self.autoscale()
            self.update_lut_line()

    def _link_layer(self, layer):

        print("qt linking layer")
        self.blockSignals(True)
        self.clims = layer.contrast_limits
        self.gamma = layer.gamma
        self.blockSignals(False)

        def set_self_gamma(x):
            self.blockSignals(True)
            self.gamma = layer.gamma
            self.blockSignals(False)

        def set_layer_gamma(x):
            with layer.events.gamma.blocker(set_self_gamma):
                layer.gamma = x

        def set_self_clims(x):
            self.blockSignals(True)
            self.clims = layer.contrast_limits
            self.blockSignals(False)

        def set_layer_clims(x):
            with layer.events.contrast_limits.blocker(set_self_clims):
                layer.contrast_limits = x

        self.gamma_updated.connect(set_layer_gamma)
        layer.events.gamma.connect(set_self_gamma)
        self.clims_updated.connect(set_layer_clims)
        layer.events.contrast_limits.connect(set_self_clims)

        self.hist_layer.link_layer(layer)

    def unlink_layer(self):
        pass

    def autoscale(self):
        e = self.hist_layer.model.bin_edges
        counts = self.hist_layer.model.counts
        x = (e[0], e[-1]) if e is not None else None
        y = (0, counts.max()) if counts is not None else None
        self.camera.set_range(x=x, y=y, margin=0.005)

    def update_lut_line(self):
        npoints = 255
        y1 = self.range[1] * 0.99
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
                self.clim_handles[i].set_data(value[i])
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

    def _to_window_coords(self, pos):
        x, y, _, _ = self.node_tform.imap(pos)
        return x, y

    def _to_plot_coords(self, pos):
        x, y, _, _ = self.node_tform.map(pos)
        return x, y

    def on_mouse_press(self, event):
        if event.pos is None:
            return
        # determine if a clim_handle was clicked
        self._clim_handle_grabbed = self._pos_is_clim(event)
        self._gamma_handle_grabbed = self._pos_is_gamma(event)
        if self._clim_handle_grabbed or self._gamma_handle_grabbed:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.camera._viewbox_unset(self.camera.viewbox)

    def on_resize(self, event):
        self.node_tform = self.plot.node_transform(self.plot.view.scene)

    def on_mouse_release(self, event):
        self._clim_handle_grabbed = 0
        self._gamma_handle_grabbed = 0
        if not self.camera.viewbox:
            self.camera._viewbox_set(self.plot.view)

    def _pos_is_clim(self, event, tolerance=3):
        """Returns 1 if x is near clims[0], 2 if near clims[1], else 0

        event is expected to to have an attribute 'pos' giving the mouse
        position be in window coordinates.
        """
        # checking clim1 first since it's more likely
        x = event.pos[0]
        clim1, _ = self._to_window_coords((self.clims[1],))
        if abs(clim1 - x) < tolerance:
            return 2

        clim0, _ = self._to_window_coords((self.clims[0],))
        if abs(clim0 - x) < tolerance:
            return 1
        return 0

    def _pos_is_gamma(self, event, tolerance=4):
        """Returns True if value is near the gamma handle.

        event is expected to to have an attribute 'pos' giving the mouse
        position be in window coordinates.
        """

        gx, gy = self._to_window_coords(self.gamma_handle._data[0][0])
        x, y = event.pos
        if abs(gx - x) < tolerance and abs(gy - y) < tolerance:
            return True
        return False

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return

        # event.pos == (0,0) is top left corner of window

        if self._clim_handle_grabbed:
            newlims = list(self.clims)
            x = self._to_plot_coords(event.pos)[0]
            newlims[self._clim_handle_grabbed - 1] = x
            self.clims = newlims
            return

        if self._gamma_handle_grabbed:
            y0, y1 = self.range
            y = self._to_plot_coords(event.pos)[1]
            if y < np.maximum(y0, 0) or y > y1:
                return
            self.gamma = -np.log2(y / y1)
            return

        QGuiApplication.restoreOverrideCursor()

        if self._pos_is_clim(event):
            QGuiApplication.setOverrideCursor(Qt.CursorShape.SplitHCursor)
        elif self._pos_is_gamma(event):
            QGuiApplication.setOverrideCursor(Qt.CursorShape.SplitVCursor)
        else:
            x, y = self._to_plot_coords(event.pos)
            x1, x2 = self.domain
            y1, y2 = self.range
            if (x1 < x <= x2) and (y1 <= y <= y2):
                QGuiApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)
