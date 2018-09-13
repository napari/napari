from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QSlider, QMenuBar

from vispy.scene import SceneCanvas

from .image_container import ImageContainer
from .panzoom import PanZoomCamera

from ..util import is_multichannel


class ImageViewerWidget(QWidget):
    """Image-based PyQt5 widget.

    Parameters
    ----------
    image : np.ndarray
        Image contained by the widget.
    meta : dict
        Image metadata.
    containing_window : PyQt5.QWindow, optional
        Window that contains the widget.
    """
    def __init__(self, image, meta, parent=None):
        super().__init__(parent=parent)

        self.image = image
        self.meta = meta

        self.point = [0] * image.ndim
        self.axis0 = 0
        self.axis1 = 1
        self.slider_index_map = {}

        layout = QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setColumnStretch(0, 4)

        row = 0

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.view = self.canvas.central_widget.add_view()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        # view.camera.zoom(0.1, (250, 200))

        self.containers = [ImageContainer(image, meta, self.view,
                                          self.canvas.update)]

        layout.addWidget(self.canvas.native, row, 0)
        layout.setRowStretch(row, 1)
        row += 1

        for axis in range(image.ndim - is_multichannel(meta)):
            if axis != self.axis0 and axis != self.axis1:
                self.add_slider(layout, row, axis, image.shape[axis])

                layout.setRowStretch(row, 4)
                row += 1

        self.update_images()

    @property
    def max_dims(self):
        """int: Maximum tunable dimensions for contained images.
        """
        max_dims = 0

        for container in self.containers:
            dims = container.image.ndim - is_multichannel(container.meta)
            max_dims = max(max_dims, dims)

        return max_dims

    @property
    def max_shape(self):
        """tuple: Maximum shape for contained images.
        """
        max_shape = tuple()

        for dim in range(self.max_dims):
            max_dim_len = 0

            for container in self.containers:
                dim_len = container.image.shape[dim]
                max_dim_len = max(max_dim_len, dim_len)

            max_shape += (max_dim_len,)

        return max_shape

    def add_slider(self, grid, row,  axis, length):
        """Adds a slider to the given grid.

        Parameters
        ----------
        grid : PyQt5.QGridLayout
            Grid layout to add the slider to.
        row : int
            Row in which to add the slider.
        axis : int
            Axis that this slider controls.
        length : int
            Maximum length of the slider.
        """
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setMinimum(0)
        slider.setMaximum(length-1)
        slider.setFixedHeight(17)
        slider.setTickPosition(QSlider.NoTicks)
        # tick_interval = int(max(8,length/8))
        # slider.setTickInterval(tick_interval)
        slider.setSingleStep(1)
        grid.addWidget(slider, row, 0)

        def value_changed():
            self.point[axis] = slider.value()
            self.update_images()

        slider.valueChanged.connect(value_changed)

    def update_images(self):
        """Updates the contained images.
        """
        indices = list(self.point)
        indices[self.axis0] = slice(None)
        indices[self.axis1] = slice(None)

        for container in self.containers:
            container.set_view(indices)
