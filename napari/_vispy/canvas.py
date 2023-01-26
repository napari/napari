"""VispyCanvas class.
"""
from weakref import WeakSet

import numpy as np
from vispy.scene import SceneCanvas, Widget

from napari._vispy import VispyCamera
from napari._vispy.utils.gl import get_max_texture_sizes
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.interactions import (
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)


class VispyCanvas:
    """SceneCanvas for our QtViewer class.

    Add two features to SceneCanvas. Ignore mousewheel events with
    modifiers, and get the max texture size in __init__().

    Attributes
    ----------
    max_texture_sizes : Tuple[int, int]
        The max textures sizes as a (2d, 3d) tuple.

    """

    _instances = WeakSet()

    def __init__(self, *args, **kwargs):

        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self.max_texture_sizes = None
        self._last_theme_color = None
        self._background_color_override = None
        self.viewer = kwargs["parent"].viewer
        self.scene_canvas = SceneCanvas(*args, **kwargs)
        self.view = self.central_widget.add_view(border_width=0)
        self.vispy_camera = VispyCamera(
            self.view, self.viewer.camera, self.viewer.dims
        )
        self.layer_to_visual = {}
        self._instances.add(self)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        self.scene_canvas.events.ignore_callback_errors = False
        self.scene_canvas.context.set_depth_func('lequal')

        # Connecting events from SceneCanvas
        self.scene_canvas.events.draw.connect(self.viewer.dims.enable_play)
        self.scene_canvas.events.draw.connect(self.vispy_camera.on_draw)

        self.scene_canvas.events.mouse_double_click.connect(
            self.on_mouse_double_click
        )
        self.scene_canvas.events.mouse_move.connect(self.on_mouse_move)
        self.scene_canvas.events.mouse_press.connect(self.on_mouse_press)
        self.scene_canvas.events.mouse_release.connect(self.on_mouse_release)
        self.scene_canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.scene_canvas.events.resize.connect(self.on_resize)
        self.scene_canvas.events.draw.connect(self.on_draw)
        self.viewer.events.theme.connect(self._on_theme_change)
        self.destroyed.connect(self._disconnect_theme)

    @property
    def destroyed(self):
        return self.scene_canvas._backend.destroyed

    @property
    def background_color_override(self):
        return self._background_color_override

    @background_color_override.setter
    def background_color_override(self, value):
        self._background_color_override = value
        self.bgcolor = value or self._last_theme_color

    def _on_theme_change(self, event):
        self._set_theme_change(event.value)

    def _set_theme_change(self, theme: str):
        from napari.utils.theme import get_theme

        # Note 1. store last requested theme color, in case we need to reuse it
        # when clearing the background_color_override, without needing to
        # keep track of the viewer.
        # Note 2. the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        self._last_theme_color = transform_color(
            get_theme(theme, False).canvas.as_hex()
        )[0]
        self.bgcolor = self._last_theme_color

    def _disconnect_theme(self):
        self.viewer.events.theme.disconnect(self._on_theme_change)

    @property
    def bgcolor(self):
        return self.scene_canvas.bgcolor.hex

    @bgcolor.setter
    def bgcolor(self, value):
        _value = self._background_color_override or value
        self.scene_canvas.bgcolor = _value

    @property
    def central_widget(self):
        """Overrides SceneCanvas.central_widget to make border_width=0"""
        if self.scene_canvas._central_widget is None:
            self.scene_canvas._central_widget = Widget(
                size=self.scene_canvas.size,
                parent=self.scene_canvas.scene,
                border_width=0,
            )
        return self.scene_canvas._central_widget

    # TODO: ask regarding functionality of this as it does not seem to be
    #  called anywhere
    # def _process_mouse_event(self, event):
    #
    #     """Ignore mouse wheel events which have modifiers."""
    #     if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
    #         return
    #     super().scene_canvas._process_mouse_event(event)

    def _map_canvas2world(self, position):
        """Map position from canvas pixels into world coordinates.

        Parameters
        ----------
        position : 2-tuple
            Position in canvas (x, y).

        Returns
        -------
        coords : tuple
            Position in world coordinates, matches the total dimensionality
            of the viewer.
        """
        nd = self.viewer.dims.ndisplay
        transform = self.view.scene.transform
        mapped_position = transform.imap(list(position))[:nd]
        position_world_slice = mapped_position[::-1]

        # handle position for 3D views of 2D data
        nd_point = len(self.viewer.dims.point)
        if nd_point < nd:
            position_world_slice = position_world_slice[-nd_point:]

        position_world = list(self.viewer.dims.point)
        for i, d in enumerate(self.viewer.dims.displayed):
            position_world[d] = position_world_slice[i]

        return tuple(position_world)

    def _process_mouse_event(self, mouse_callbacks, event):
        """Add properties to the mouse event before passing the event to the
        napari events system. Called whenever the mouse moves or is clicked.
        As such, care should be taken to reduce the overhead in this function.
        In future work, we should consider limiting the frequency at which
        it is called.

        This method adds following:
            position: the position of the click in world coordinates.
            view_direction: a unit vector giving the direction of the camera in
                world coordinates.
            up_direction: a unit vector giving the direction of the camera that is
                up in world coordinates.
            dims_displayed: a list of the dimensions currently being displayed
                in the viewer. This comes from viewer.dims.displayed.
            dims_point: the indices for the data in view in world coordinates.
                This comes from viewer.dims.point

        Parameters
        ----------
        mouse_callbacks : function
            Mouse callbacks function.
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        if event.pos is None:
            return

        # Add the view ray to the event
        event.view_direction = self.viewer.camera.calculate_nd_view_direction(
            self.viewer.dims.ndim, self.viewer.dims.displayed
        )
        event.up_direction = self.viewer.camera.calculate_nd_up_direction(
            self.viewer.dims.ndim, self.viewer.dims.displayed
        )

        # Update the cursor position
        self.viewer.cursor._view_direction = event.view_direction
        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))

        # Add the cursor position to the event
        event.position = self.viewer.cursor.position

        # Add the displayed dimensions to the event
        event.dims_displayed = list(self.viewer.dims.displayed)

        # Add the current dims indices
        event.dims_point = list(self.viewer.dims.point)

        # Put a read only wrapper on the event
        event = ReadOnlyWrapper(event)
        mouse_callbacks(self.viewer, event)

        layer = self.viewer.layers.selection.active
        if layer is not None:
            mouse_callbacks(layer, event)

    def on_mouse_double_click(self, event):
        """Called whenever a mouse double-click happen on the canvas

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method. The `event.type` will always be `mouse_double_click`

        Notes
        -----

        Note that this triggers in addition to the usual mouse press and mouse release.
        Therefore a double click from the user will likely triggers the following event in sequence:

             - mouse_press
             - mouse_release
             - mouse_double_click
             - mouse_release
        """
        self._process_mouse_event(mouse_double_click_callbacks, event)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_move_callbacks, event)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_press_callbacks, event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_release_callbacks, event)

    def on_mouse_wheel(self, event):
        """Called whenever mouse wheel activated in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_wheel_callbacks, event)

    @property
    def _canvas_corners_in_world(self):
        """Location of the corners of canvas in world coordinates.

        Returns
        -------
        corners : 2-tuple
            Coordinates of top left and bottom right canvas pixel in the world.
        """
        # Find corners of canvas in world coordinates
        top_left = self._map_canvas2world([0, 0])
        bottom_right = self._map_canvas2world(self.viewer._canvas_size[::-1])
        return np.array([top_left, bottom_right])

    def on_draw(self, event):
        """Called whenever the canvas is drawn.

        This is triggered from vispy whenever new data is sent to the canvas or
        the camera is moved and is connected in the `QtViewer`.
        """
        # The canvas corners in full world coordinates (i.e. across all layers).
        canvas_corners_world = self._canvas_corners_in_world
        for layer in self.viewer.layers:
            # The following condition should mostly be False. One case when it can
            # be True is when a callback connected to self.viewer.dims.events.ndisplay
            # is executed before layer._slice_input has been updated by another callback
            # (e.g. when changing self.viewer.dims.ndisplay from 3 to 2).
            displayed_sorted = sorted(layer._slice_input.displayed)
            nd = len(displayed_sorted)
            if nd > self.viewer.dims.ndisplay:
                displayed_axes = displayed_sorted
            else:
                displayed_axes = self.viewer.dims.displayed[-nd:]
            layer._update_draw(
                scale_factor=1 / self.viewer.camera.zoom,
                corner_pixels_displayed=canvas_corners_world[
                    :, displayed_axes
                ],
                shape_threshold=self.viewer._canvas_size,
            )

    def on_resize(self, event):
        """Called whenever canvas is resized.

        event : vispy.util.event.Event
            The vispy event that triggered this method.
        """
        self.viewer._canvas_size = tuple(self.scene_canvas.size[::-1])

    def add_layer_to_visual(self, napari_layer, vispy_layer):
        if not self.viewer.grid.enabled:
            vispy_layer.node.parent = self.view.scene
            self.layer_to_visual[napari_layer] = vispy_layer
