from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QSlider, QMenuBar

from vispy.scene import SceneCanvas

from .image_container import ImageContainer
from .panzoom import PanZoomCamera

from ..util import is_multichannel, compute_max_shape as _compute_max_shape


class ImageViewerWidget(QWidget):
    """Image-based PyQt5 widget.

    Parameters
    ----------
    parent : PyQt5.QWidget, optional
        Parent widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # TODO: allow arbitrary display axis setting
        # self.y_axis = 0  # typically the y-axis
        # self.x_axis = 1  # typically the x-axis
        self.point = []
        self.containers = []
        self.sliders = []

        layout = QGridLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0);
        layout.setColumnStretch(0, 4)

        self.canvas = SceneCanvas(keys=None, vsync=True)

        row = 0
        layout.addWidget(self.canvas.native, row, 0)
        layout.setRowStretch(row, 1)

        self.view = self.canvas.central_widget.add_view()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        # view.camera.zoom(0.1, (250, 200))

    def _axis_to_row(self, axis):
        dims = len(self.point)
        message = f'axis {axis} out of bounds for {dims} dims'

        if axis < 0:
            axis = dims - axis
            if axis < 0:
                raise IndexError(message)
        elif axis >= dims:
            raise IndexError(message)

        if axis < 2:
            raise ValueError('cannot convert y/x-axes to rows')

        return axis - 1

    def add_image(self, image, meta):
        """Adds an image to the viewer.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict
            Image metadata.

        Returns
        -------
        container : ImageContainer
            Container for the image.
        """

        container = ImageContainer(image, meta, self.view)
        self.containers.append(container)

        self.update_sliders()
        self.update_images()

        return container

    def update_sliders(self):
        """Updates the sliders according to the contained images.
        """
        max_shape = self.max_shape
        max_dims = len(max_shape)

        curr_dims = len(self.point)

        if curr_dims > max_dims:
            self.point = self.point[:max_dims]
            dims = curr_dims
        else:
            dims = max_dims
            self.point.extend([0] * (max_dims - curr_dims))

        for dim in range(2, dims):  # do not create sliders for y/x-axes
            try:
                dim_len = max_shape[dim]
            except IndexError:
                dim_len = 0

            self.update_slider(dim, dim_len)

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
        row = self._axis_to_row(axis)

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
                self.point[axis] = slider.value()
                self.update_images()

            slider.valueChanged.connect(value_changed)

            grid.addWidget(slider, row, 0)
            self.sliders.append(slider)

        slider.setMaximum(max_axis_length - 1)
        return slider

    def update_images(self):
        """Updates the contained images.
        """
        indices = list(self.point)
        indices[0] = slice(None)  # y-axis
        indices[1] = slice(None)  # x-axis

        for container in self.containers:
            container.set_view_slice(indices)

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
        shapes = (container.image.shape for container in self.containers)
        max_shape = _compute_max_shape(shapes, self.max_dims)

        return max_shape
