from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QGridLayout, QSlider, QVBoxLayout

from vispy.scene import SceneCanvas, PanZoomCamera

import weakref


class QtViewer(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = weakref.proxy(viewer)
        self.sliders = []

        self.canvas = SceneCanvas(keys=None, vsync=True)

        layout = QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        #layout.setColumnStretch(0, 4)

        row = 0
        layout.addWidget(self.canvas.native, row, 0)
        #layout.setRowStretch(row, 1)

        self.view = self.canvas.central_widget.add_view()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)

        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        # view.camera.zoom(0.1, (250, 200))

    def update_slider(self, axis, max_axis_length):
        """Updates a slider for the given axis or creates
        it if it does not already exist.

        Parameters
        ----------
        axis : int
            Axis that this slider controls.
        max_axis_length : int
            Longest length for this axis. If 0, deletes the slider.

        Returns
        -------
        slider : PyQt5.QSlider or None
            Updated slider, if it exists.
        """
        grid = self.layout()
        row = self.viewer._axis_to_row(axis)

        slider = grid.itemAt(row)
        if max_axis_length <= 0:
            # delete slider
            grid.takeAt(row)
            return

        if slider is None:  # has not been created yet
            # create slider
            if axis < 0:
                raise ValueError('cannot create a slider '
                                 f'at negative axis {axis}')

            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setMinimum(0)
            slider.setFixedHeight(17)
            slider.setTickPosition(QSlider.NoTicks)
            # slider.setTickPosition(QSlider.TicksBothSides)
            # tick_interval = int(max(8,max_axis_length/8))
            # slider.setTickInterval(tick_interval)
            slider.setSingleStep(1)

            def value_changed():
                self.viewer.indices[axis] = slider.value()
                self.viewer._need_redraw = True
                self.viewer._update()

            slider.valueChanged.connect(value_changed)

            grid.addWidget(slider, row, 0)
            self.sliders.append(slider)
        else:
            slider = slider.widget()

        slider.setMaximum(max_axis_length - 1)
        return slider
