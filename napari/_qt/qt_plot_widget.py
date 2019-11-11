from qtpy.QtCore import QSize
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from vispy import scene

from .._vispy.vispy_plot import NapariPlotWidget


class QtPlotWidget(QWidget):
    def __init__(self, vertical=False, parent=None):
        super().__init__(parent)

        self.vertical = vertical

        self.canvas = scene.SceneCanvas(bgcolor='k', keys=None, vsync=True)
        self.canvas.events.ignore_callback_errors = False
        if vertical:
            self.canvas.native.setMinimumSize(QSize(100, 300))
            self.canvas.native.resize(200, 800)
        else:
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
                show_yaxis=vertical,
                show_xaxis=not vertical,
                lock_axis=(0 if vertical else 1),
            )
        )
        self.plot._configure_2d()
        self.camera = self.plot.view.camera
        self.camera.set_range(margin=0.005)

    def on_resize(self, event):
        self.node_tform = self.plot.node_transform(self.plot.view.scene)

    @property
    def domain(self):
        if self.vertical:
            return self.plot.yaxis.axis.domain
        else:
            return self.plot.xaxis.axis.domain

    @property
    def range(self):
        if self.vertical:
            return self.plot.xaxis.axis.domain
        else:
            return self.plot.yaxis.axis.domain

    def _to_window_coords(self, pos):
        x, y, _, _ = self.node_tform.imap(pos)
        return x, y

    def _to_plot_coords(self, pos):
        x, y, _, _ = self.node_tform.map(pos)
        return x, y
