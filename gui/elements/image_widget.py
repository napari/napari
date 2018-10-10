from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QSlider, QMenuBar

from vispy.scene import SceneCanvas, PanZoomCamera

from .image_container import ImageContainer

from .layouts import HorizontalLayout, VerticalLayout, StackedLayout

from ..util.misc import (compute_max_shape as _compute_max_shape,
                         guess_metadata)


class ImageViewerWidget(QWidget):
    """Image-based PyQt5 widget.

    Parameters
    ----------
    parent : PyQt5.QWidget, optional
        Parent window.
    """
    layout_map = {
        'horizontal': HorizontalLayout,
        'vertical': VerticalLayout,
        'stacked': StackedLayout
    }

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # TODO: allow arbitrary display axis setting
        # self.y_axis = 0  # typically the y-axis
        # self.x_axis = 1  # typically the x-axis
        self.point = []
        self.containers = []
        self.containerlayout = HorizontalLayout(self)

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

        self._max_dims = 0
        self._max_shape = tuple()

        # update flags
        self._child_image_changed = False
        self._need_redraw = False
        self._need_slider_update = False

        self._recalc_max_dims = False
        self._recalc_max_shape = False

    @property
    def layout_type(self):
        """str: Layout display type.
        """
        for name, layout in self.layout_map.items():
            if isinstance(self.containerlayout, layout):
                return name
        raise Exception()

    @layout_type.setter
    def layout_type(self, layout):
        if layout == self.layout_type:
            return

        layout = self.layout_map[layout].from_layout(self.containerlayout)
        self.containerlayout = layout
        self.reset_view()

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
        meta : dict, optional
            Image metadata.

        Returns
        -------
        container : ImageContainer
            Container for the image.
        """
        container = ImageContainer(image, meta, self)

        self.containers.append(container)
        self.containerlayout.add_container(container)

        self._child_image_changed = True
        self.update()

        return container

    def imshow(self, image, meta=None, multichannel=None, **kwargs):
        """Shows an image in the viewer.

        Parameters
        ----------
        image : np.ndarray
            Image data.
        meta : dict, optional
            Image metadata.
        multichannel : bool, optional
            Whether the image is multichannel. Guesses if None.
        **kwargs : dict
            Parameters that will be translated to metadata.

        Returns
        -------
        container : ImageContainer
            Container for the image.
        """
        meta = guess_metadata(image, meta, multichannel, kwargs)

        return self.add_image(image, meta)

    def reset_view(self):
        """Resets the camera's view.
        """
        try:
            self.view.camera.set_range(*self.containerlayout.view_range)
        except AttributeError:
            pass

    def _update_slider(self, axis, max_axis_length):
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
                self._need_redraw = True
                self.update()

            slider.valueChanged.connect(value_changed)

            grid.addWidget(slider, row, 0)
            self.sliders.append(slider)
        else:
            slider = slider.widget()

        slider.setMaximum(max_axis_length - 1)
        return slider

    def _update_sliders(self):
        """Updates the sliders according to the contained images.
        """
        max_dims = self.max_dims
        max_shape = self.max_shape

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

            self._update_slider(dim, dim_len)

    def _update_images(self):
        """Updates the contained images.
        """
        for container in self.containers:
            container.set_view_slice(self.point)

    def _calc_max_dims(self):
        """Calculates the number of maximum dimensions in the contained images.
        """
        max_dims = 0

        for container in self.containers:
            dims = container.effective_ndim
            if dims > max_dims:
                max_dims = dims

        self._max_dims = max_dims

    def _calc_max_shape(self):
        """Calculates the maximum shape of the contained images.
        """
        shapes = (container.image.shape for container in self.containers)
        self._max_shape = _compute_max_shape(shapes, self.max_dims)

    def update(self):
        """Updates the viewer.
        """
        if self._child_image_changed:
            self._child_image_changed = False
            self._recalc_max_dims = True
            self._recalc_max_shape = True
            self._need_slider_update = True

            self.containerlayout.update()
            self.reset_view()

        if self._need_redraw:
            self._need_redraw = False
            self._update_images()

        if self._recalc_max_dims:
            self._recalc_max_dims = False
            self._calc_max_dims()

        if self._recalc_max_shape:
            self._recalc_max_shape = False
            self._calc_max_shape()

        if self._need_slider_update:
            self._need_slider_update = False
            self._update_sliders()

    @property
    def max_dims(self):
        """int: Maximum tunable dimensions for contained images.
        """
        return self._max_dims

    @property
    def max_shape(self):
        """tuple: Maximum shape for contained images.
        """
        return self._max_shape
