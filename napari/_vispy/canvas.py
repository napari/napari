"""VispyCanvas class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakSet

import numpy as np
from vispy.scene import SceneCanvas as SceneCanvas_
from vispy.scene import Widget

from napari._vispy import VispyCamera
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.utils.visual import create_vispy_overlay
from napari.components.overlays import CanvasOverlay, SceneOverlay
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.interactions import (
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)

if TYPE_CHECKING:
    from qtpy.QtGui import QCursor, QImage

    from napari.components import ViewerModel
    from napari.components.overlays import Overlay


class SceneCanvas(SceneCanvas_):
    """Vispy SceneCanvas used to allow for ignoring mouse wheel events with modifiers."""

    def _process_mouse_event(self, event):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        super()._process_mouse_event(event)


class VispyCanvas:
    """Class for our QtViewer class to interact with Vispy SceneCanvas. Also
    connects Vispy SceneCanvas events to the napari ViewerModel and vice versa.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    max_texture_sizes : Tuple[int, int]
        The max textures sizes as a (2d, 3d) tuple.
    last_theme_color : Optional[npt.NDArray[np.float]]
        Theme color represented as numpy ndarray of shape (4,) before theme change
        was applied.
    background_color_override : Optional[npt.NDArray[np.float]]
        Background color of the canvas represented as numpy ndarray of shape (4,) which overrides the
        canvas background color indicated in the napari theme settings.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    scene_canvas : vispy.scene.SceneCanvas
        The vispy SceneCanvas which automatically draws the contents of a scene.
    view : vispy.scene.widgets.viewbox.ViewBox
        Rectangular widget in which a subscene is rendered.
    vispy_camera : napari._vispy.VispyCamera
        The camera class which contains both the 2d and 3d camera used to describe the perspective by which a
        scene is viewed and interacted with.
    layer_to_visual : dict(napari.layers, napari._vispy.layers)
        A mapping of the napari layers that have been added to the viewer and their corresponding vispy counterparts.
    """

    _instances = WeakSet()

    def __init__(self, viewer: ViewerModel, *args, **kwargs) -> None:

        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self.max_texture_sizes = None
        self._last_theme_color = None
        self._background_color_override = None
        self.viewer = viewer
        self._scene_canvas = SceneCanvas(
            *args, keys=None, vsync=True, **kwargs
        )
        self.view = self.central_widget.add_view(border_width=0)
        self._vispy_camera = VispyCamera(
            self.view, self.viewer.camera, self.viewer.dims
        )
        self._layer_to_visual = {}
        self._overlay_to_visual = {}
        self._instances.add(self)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        for overlay in self.viewer._overlays.values():
            self._add_overlay_to_visual(overlay)

        self._scene_canvas.events.ignore_callback_errors = False
        self._scene_canvas.context.set_depth_func('lequal')

        # Connecting events from SceneCanvas
        self._scene_canvas.events.draw.connect(self.enable_dims_play)
        self._scene_canvas.events.draw.connect(self._vispy_camera.on_draw)

        self._scene_canvas.events.mouse_double_click.connect(
            self._on_mouse_double_click
        )
        self._scene_canvas.events.mouse_move.connect(self._on_mouse_move)
        self._scene_canvas.events.mouse_press.connect(self._on_mouse_press)
        self._scene_canvas.events.mouse_release.connect(self._on_mouse_release)
        self._scene_canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self._scene_canvas.events.resize.connect(self.on_resize)
        self._scene_canvas.events.draw.connect(self.on_draw)
        self.viewer.events.theme.connect(self._on_theme_change)
        self.viewer.camera.events.interactive.connect(self._on_interactive)
        self.viewer.layers.events.reordered.connect(self._reorder_layers)
        self.viewer.layers.events.removed.connect(self._remove_layer)
        self.destroyed.connect(self._disconnect_theme)

    @property
    def destroyed(self):
        return self._scene_canvas._backend.destroyed

    @property
    def native(self):
        """Returns the native widget of the Vispy SceneCanvas."""
        return self._scene_canvas.native

    @property
    def screen_changed(self):
        """Returning signal indicating whether the window screen has changed."""
        return self._scene_canvas._backend.screen_changed

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
        return self._scene_canvas.bgcolor.hex

    @bgcolor.setter
    def bgcolor(self, value):
        _value = self._background_color_override or value
        self._scene_canvas.bgcolor = _value

    @property
    def central_widget(self):
        """Overrides SceneCanvas.central_widget to make border_width=0"""
        if self._scene_canvas._central_widget is None:
            self._scene_canvas._central_widget = Widget(
                size=self.size,
                parent=self._scene_canvas.scene,
                border_width=0,
            )
        return self._scene_canvas._central_widget

    @property
    def size(self):
        """Return canvas size as tuple (height, width) or accepts size as tuple (height, width)
        and sets Vispy SceneCanvas size as (width, height)."""
        return self._scene_canvas.size[::-1]

    @size.setter
    def size(self, size):
        self._scene_canvas.size = size[::-1]

    @property
    def cursor(self) -> QCursor:
        """Cursor associated with native widget"""
        return self.native.cursor()

    @cursor.setter
    def cursor(self, q_cursor):
        """Setting the cursor of the native widget"""
        self.native.setCursor(q_cursor)

    def delete(self):
        """Schedules the native widget for deletion"""
        self.native.deleteLater()

    def _on_interactive(self):
        """Link interactive attributes of view and viewer."""
        self.view.interactive = self.viewer.camera.interactive

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

    def _on_mouse_double_click(self, event):
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

    def _on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_move_callbacks, event)

    def _on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_press_callbacks, event)

    def _on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_release_callbacks, event)

    def _on_mouse_wheel(self, event):
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
        bottom_right = self._map_canvas2world(self._scene_canvas.size)
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
                shape_threshold=self._scene_canvas.size,
            )

    def on_resize(self, event):
        """Called whenever canvas is resized.

        event : vispy.util.event.Event
            The vispy event that triggered this method.
        """
        self.viewer._canvas_size = tuple(self.size)

    def add_layer_to_visual(self, napari_layer, vispy_layer):
        if not self.viewer.grid.enabled:
            vispy_layer.node.parent = self.view.scene
            self._layer_to_visual[napari_layer] = vispy_layer
        self._reorder_layers()

    def _remove_layer(self, event):
        layer = event.value
        vispy_layer = self._layer_to_visual[layer]
        vispy_layer.close()
        del vispy_layer
        del self._layer_to_visual[layer]
        self._reorder_layers()

    def _reorder_layers(self):
        """When the list is reordered, propagate changes to draw order."""
        for i, layer in enumerate(self.viewer.layers):
            vispy_layer = self._layer_to_visual[layer]
            vispy_layer.order = i
        self._scene_canvas._draw_order.clear()
        self._scene_canvas.update()

    def _add_overlay_to_visual(self, overlay: Overlay):
        """Create vispy overlay and add to dictionary of overlay visuals"""
        # TODO: Fix issue with node.parent.parent not having attribute background_color_override.
        vispy_overlay = create_vispy_overlay(overlay, viewer=self.viewer)
        if isinstance(overlay, CanvasOverlay):
            vispy_overlay.node.parent = self.view
        elif isinstance(overlay, SceneOverlay):
            vispy_overlay.node.parent = self.view.scene
        self._overlay_to_visual[overlay] = vispy_overlay

    def screenshot(self) -> QImage:
        """Return a QImage based on what is shown in the viewer."""
        return self.native.grabFramebuffer()

    def enable_dims_play(self, *args):
        """Enable playing of animation. False if awaiting a draw event"""
        self.viewer.dims._play_ready = True
