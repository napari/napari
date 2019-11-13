import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QScrollBar
from vispy import scene

from .._vispy.vispy_histogram import VispyHistogramLayer
from .qt_plot_widget import QtPlotWidget
from ..util.misc import blocked_qt_signals


class QtHistogramWidget(QtPlotWidget):

    clims_updated = Signal(tuple)
    gamma_updated = Signal(float)

    def __init__(
        self,
        layer=None,
        viewer=None,
        clims=(0, 255),
        gamma=1,
        clim_handle_color=(0.26, 0.28, 0.31, 1),
        vertical=False,
        link='view',
    ):
        super().__init__(viewer=viewer, vertical=vertical)

        self.hist_layer = VispyHistogramLayer(
            link=link, orientation='v' if vertical else 'h'
        )

        self._clims = clims
        self._gamma = gamma
        self._layer = None

        self.gamma_handle = None
        self.lut_line = None
        self._gamma_handle_position = None
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
        self.update_lut_lines()

        slider = QScrollBar(Qt.Vertical)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(2)
        slider.setMaximum(20)
        slider.setValue(2)
        slider.valueChanged.connect(self.on_logslider_change)
        slider.setInvertedAppearance(True)
        self.layout.insertWidget(0, slider)

    def on_logslider_change(self, value):
        self.hist_layer.model.pow = np.log(5) / value

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
            self.update_lut_lines()

    def _get_layer_gamma(self, *args):
        with blocked_qt_signals(self):
            self.gamma = self.layer.gamma

    def _set_layer_gamma(self, value):
        with self.layer.events.gamma.blocker(self._get_layer_gamma):
            self.layer.gamma = value

    def _get_layer_clims(self, *args):
        with blocked_qt_signals(self):
            self.clims = self.layer.contrast_limits

    def _set_layer_clims(self, value):
        with self.layer.events.contrast_limits.blocker(self._get_layer_clims):
            self.layer.contrast_limits = value

    def _link_layer(self, layer):
        with blocked_qt_signals(self):
            self.clims = layer.contrast_limits
            self.gamma = layer.gamma

        self.gamma_updated.connect(self._set_layer_gamma)
        layer.events.gamma.connect(self._get_layer_gamma)
        self.clims_updated.connect(self._set_layer_clims)
        layer.events.contrast_limits.connect(self._get_layer_clims)
        self.hist_layer.link_layer(layer)

    def unlink_layer(self):
        self.gamma_updated.disconnect(self._set_layer_gamma)
        self.layer.events.gamma.disconnect(self._get_layer_gamma)
        self.clims_updated.disconnect(self._set_layer_clims)
        self.layer.events.contrast_limits.disconnect(self._get_layer_clims)

    def autoscale(self):
        e = self.hist_layer.model.bin_edges
        counts = self.hist_layer.model.counts
        if self.vertical:
            y = (e[0], e[-1]) if e is not None else None
            x = (0, counts.max()) if counts is not None else None
        else:
            x = (e[0], e[-1]) if e is not None else None
            y = (0, counts.max()) if counts is not None else None
        self.camera.set_range(x=x, y=y, margin=0.005)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not np.isscalar(value):
            raise ValueError('gamma value must be a scalar')
        self._gamma = float(value)
        self.update_lut_lines()
        self.gamma_updated.emit(value)

    @property
    def clims(self):
        return self._clims

    @clims.setter
    def clims(self, value):
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
            raise ValueError('clims value must be a 2-item array')
        self._clims = value
        self.update_lut_lines()
        self.clims_updated.emit(value)

    def update_lut_lines(self, npoints=256):
        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)
        y1 = self.range[1] * 0.98
        if self.vertical:
            # clims lines
            X[0:2], Y[0:2] = (y1, y1 / 2), self.clims[0]
            X[-2:], Y[-2:] = (y1 / 2, 0), self.clims[1]
            # gamma line
            X[2:-2] = np.linspace(0, 1, npoints) ** self.gamma * y1
            Y[2:-2] = np.linspace(self.clims[0], self.clims[1], npoints)
            midpoint = np.array([(y1 * 2 ** -self.gamma, np.mean(self.clims))])
        else:
            # clims lines
            X[0:2], Y[0:2] = self.clims[0], (y1, y1 / 2)
            X[-2:], Y[-2:] = self.clims[1], (y1 / 2, 0)
            # gamma line
            X[2:-2] = np.linspace(self.clims[0], self.clims[1], npoints)
            Y[2:-2] = np.linspace(0, 1, npoints) ** self.gamma * y1
            midpoint = np.array([(np.mean(self.clims), y1 * 2 ** -self.gamma)])

        if self.lut_line:
            self.lut_line.set_data((X, Y), marker_size=0)
        else:

            color = np.linspace(0.2, 0.8, npoints + 4).repeat(4).reshape(-1, 4)
            c1, c2 = [0.4] * 4, [0.7] * 4
            color[0:3] = [c1, c2, c1]
            color[-3:] = [c1, c2, c1]
            self.lut_line = self.plot.plot(
                (X, Y), marker_size=0, width=1.5, color=color
            )
            self.lut_line.order = -1

        self._gamma_handle_position = midpoint[0]
        gkwargs = {'size': 6, 'edge_width': 0}
        if self.gamma_handle:
            self.gamma_handle.set_data(pos=midpoint, **gkwargs)
        else:
            self.gamma_handle = scene.Markers(pos=midpoint, **gkwargs)
            self.gamma_handle.order = -2
            self.plot.view.add(self.gamma_handle)

    def on_mouse_press(self, event):
        if event.pos is None:
            return
        # determine if a clim_handle was clicked
        self._clim_handle_grabbed = self._pos_is_clim(event)
        self._gamma_handle_grabbed = self._pos_is_gamma(event)
        if self._clim_handle_grabbed or self._gamma_handle_grabbed:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.camera._viewbox_unset(self.camera.viewbox)

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
        if self.vertical:
            x = event.pos[1]
            _, clim1 = self._to_window_coords((0, self.clims[1]))
            _, clim0 = self._to_window_coords((0, self.clims[0]))
        else:
            x = event.pos[0]
            clim1, _ = self._to_window_coords((self.clims[1],))
            clim0, _ = self._to_window_coords((self.clims[0],))
        if abs(clim1 - x) < tolerance:
            return 2
        if abs(clim0 - x) < tolerance:
            return 1
        return 0

    def _pos_is_gamma(self, event, tolerance=4):
        """Returns True if value is near the gamma handle.

        event is expected to to have an attribute 'pos' giving the mouse
        position be in window coordinates.
        """
        if self._gamma_handle_position is None:
            return False
        gx, gy = self._to_window_coords(self._gamma_handle_position)
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
            if self.vertical:
                c = self._to_plot_coords(event.pos)[1]
            else:
                c = self._to_plot_coords(event.pos)[0]
            newlims[self._clim_handle_grabbed - 1] = c
            self.clims = newlims
            return

        if self._gamma_handle_grabbed:
            y0, y1 = self.range
            y = self._to_plot_coords(event.pos)[0 if self.vertical else 1]
            if y < np.maximum(y0, 0) or y > y1:
                return
            self.gamma = -np.log2(y / y1)
            return

        self.unsetCursor()

        if self._pos_is_clim(event):
            if self.vertical:
                cursor = Qt.CursorShape.SplitVCursor
            else:
                cursor = Qt.CursorShape.SplitHCursor
            self.setCursor(cursor)
        elif self._pos_is_gamma(event):
            if self.vertical:
                cursor = Qt.CursorShape.SplitHCursor
            else:
                cursor = Qt.CursorShape.SplitVCursor
            self.setCursor(cursor)
        else:
            x, y = self._to_plot_coords(event.pos)
            x1, x2 = self.domain
            y1, y2 = self.range
            if (x1 < x <= x2) and (y1 <= y <= y2):
                self.setCursor(Qt.CursorShape.CrossCursor)
