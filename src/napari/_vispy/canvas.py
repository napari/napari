"""VispyCanvas class."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from functools import partial
from itertools import zip_longest
from typing import TYPE_CHECKING
from weakref import WeakSet

import numpy as np
from superqt.utils import qthrottled
from vispy.scene import SceneCanvas as SceneCanvas_, ViewBox, Widget

from napari._vispy.camera import VispyCamera
from napari._vispy.mouse_event import NapariMouseEvent
from napari._vispy.utils.cursor import QtCursorVisual
from napari._vispy.utils.gl import get_max_texture_sizes
from napari._vispy.utils.visual import create_vispy_overlay
from napari.components._viewer_constants import CanvasPosition
from napari.components.overlays import CanvasOverlay
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import disconnect_events
from napari.utils.events.event import Event
from napari.utils.interactions import (
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)
from napari.utils.theme import get_theme

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from qtpy.QtCore import Qt, pyqtBoundSignal
    from qtpy.QtGui import QCursor, QImage
    from vispy.app.backends._qt import CanvasBackendDesktop
    from vispy.app.canvas import DrawEvent, MouseEvent, ResizeEvent
    from vispy.scene import Node

    from napari._vispy.layers.base import VispyBaseLayer
    from napari._vispy.overlays.base import VispyBaseOverlay
    from napari.components import ViewerModel
    from napari.components.overlays import Overlay
    from napari.layers import Layer
    from napari.utils.key_bindings import KeymapHandler


import warnings

from napari.utils.translations import trans


class NapariSceneCanvas(SceneCanvas_):
    """Vispy SceneCanvas used to allow for ignoring mouse wheel events with modifiers."""

    def _process_mouse_event(self, event: MouseEvent):
        """Ignore mouse wheel events which have modifiers."""
        if event.type == 'mouse_wheel' and len(event.modifiers) > 0:
            return
        if event.handled:
            return
        super()._process_mouse_event(event)

    def draw_visual(self, visual, event=None):
        try:
            super().draw_visual(visual, event=event)
        except RuntimeError as e:
            error_msg = e.args[0] if e.args else ''
            to_ignore = (
                'Cannot draw program if code has not been set',
                'Cannot set uniform when program has no code',
            )
            if any(msg in error_msg for msg in to_ignore):
                # these always happens after another (real) error is raised, and
                # they flood the traceback because they are fired on each event,
                # hiding the actual source of the problem. They are never really
                # informative, so we can safely ignore them.
                return
            if 'Shader compilation error' in error_msg:
                raise RuntimeError(
                    'Shader compilation failed. Unless you are working with custom shader code,\n'
                    'this is likely a bug in napari.\n'
                    'Please open an issue on the repository and provide this *full* stack trace.'
                ) from e
            if 'Cannot SIZE object' in error_msg:
                raise RuntimeError(
                    'The above error may be caused by a version mismatch between vispy and napari.\n'
                    'Try recreating a fresh environment and reinstalling. If that does not work,\n'
                    'please open an issue on the repository and provide this *full* stack trace.'
                ) from e
            raise


class VispyCanvas:
    """Class for our QtViewer class to interact with Vispy SceneCanvas. Also
    connects Vispy SceneCanvas events to the napari ViewerModel and vice versa.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.

    Attributes
    ----------
    layer_to_visual : dict(napari.layers, napari._vispy.layers)
        A mapping of the napari layers that have been added to the viewer and their corresponding vispy counterparts.
    max_texture_sizes : Tuple[int, int]
        The max textures sizes as a (2d, 3d) tuple.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    view : vispy.scene.widgets.viewbox.ViewBox
        Rectangular widget in which a subscene is rendered.
    camera : napari._vispy.VispyCamera
        The camera class which contains both the 2d and 3d camera used to describe the perspective by which a
        scene is viewed and interacted with.
    _cursors : QtCursorVisual
        A QtCursorVisual enum with as names the names of particular cursor styles and as value either a staticmethod
        creating a bitmap or a Qt.CursorShape enum value corresponding to the particular cursor name. This enum only
        contains cursors supported by Napari in Vispy.
    _key_map_handler : napari.utils.key_bindings.KeymapHandler
        KeymapHandler handling the calling functionality when keys are pressed that have a callback function mapped.
    _last_theme_color : Optional[npt.NDArray[np.float]]
        Theme color represented as numpy ndarray of shape (4,) before theme change
        was applied.
    _overlay_to_visual : dict(napari.components.overlays, list(napari._vispy.overlays))
        A mapping of the napari overlays that are part of the viewer and their corresponding Vispy counterparts.
        The values are lists that may contain multiple elements when grid mode is enabled and overlay.gridded == True,
        associating multiple vispy visual to a single overlay model.
    _layer_overlay_to_visual : dict(napari.layers.Layer, dict(napari.components.overlays, napari._vispy.overlays))
        A mapping from each layer in the layerlist to their mappings of napari overlay->vispy counterpart.
    _scene_canvas : napari._vispy.canvas.NapariSceneCanvas
        SceneCanvas which automatically draws the contents of a scene. It is ultimately a VispySceneCanvas, but allows
        for ignoring mousewheel events with modifiers.
    """

    _instances: WeakSet[VispyCanvas] = WeakSet()

    def __init__(
        self,
        viewer: ViewerModel,
        key_map_handler: KeymapHandler,
        *args,
        **kwargs,
    ) -> None:
        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self._pause_scene_graph = False
        self.max_texture_sizes = None
        self._last_theme_color = None
        self._background_color_override = None
        self.viewer = viewer
        self._scene_canvas = NapariSceneCanvas(
            *args, keys=None, vsync=True, **kwargs
        )

        self.view = self.central_widget.add_view(border_width=0)
        self.view.order = 100  # ensure it's always drawn on top
        self.camera = VispyCamera(
            self.view, self.viewer.camera, self.viewer.dims
        )

        self.grid = self.central_widget.add_grid(
            border_width=0,
        )
        self.grid_views = []
        self.grid_cameras = []

        self.layer_to_visual: dict[Layer, VispyBaseLayer[Layer]] = {}
        self._overlay_to_visual: dict[Overlay, list[VispyBaseOverlay]] = {}
        self._layer_overlay_to_visual: dict[
            Layer, dict[Overlay, VispyBaseOverlay]
        ] = {}
        self._key_map_handler = key_map_handler
        self._instances.add(self)

        self._overlay_callbacks = {}
        self._last_viewbox_size = np.array((0, 0))
        self._needs_overlay_position_update = False

        self.bgcolor = transform_color(
            get_theme(self.viewer.theme).canvas.as_hex()
        )[0]

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        self._update_grid_spacing()
        self._update_viewer_overlays()

        self._scene_canvas.events.ignore_callback_errors = False
        self._scene_canvas.context.set_depth_func('lequal')

        self._scene_canvas.events.key_press.connect(
            self._key_map_handler.on_key_press
        )
        self._scene_canvas.events.key_release.connect(
            self._key_map_handler.on_key_release
        )
        self._scene_canvas.events.draw.connect(self.enable_dims_play)
        self._scene_canvas.events.mouse_double_click.connect(
            self._on_mouse_double_click
        )
        self._scene_canvas.events.mouse_move.connect(
            qthrottled(self._on_mouse_move, timeout=5)
        )
        self._scene_canvas.events.mouse_press.connect(self._on_mouse_press)
        self._scene_canvas.events.mouse_release.connect(self._on_mouse_release)
        self._scene_canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self._scene_canvas.events.resize.connect(self.on_resize)
        self._scene_canvas.events.draw.connect(self.on_draw, position='last')
        self.viewer.cursor.events.style.connect(self._on_cursor)
        self.viewer.cursor.events.size.connect(self._on_cursor)
        # position=first is important to some downstream components such as
        # scale_bar overlay which need to have access to the updated color
        # by the time they get updated as well
        self.viewer.events.theme.connect(
            self._on_theme_change, position='first'
        )

        self.viewer.camera.events.mouse_pan.connect(self._on_interactive)
        self.viewer.camera.events.mouse_zoom.connect(self._on_interactive)
        self.viewer.camera.events.zoom.connect(self._on_cursor)

        self.viewer._zoom_box.events.zoom.connect(self._on_boxzoom)
        self.viewer.layers.events.reordered.connect(self._update_scenegraph)
        self.viewer.layers.events.removed.connect(self._remove_layer)
        self.viewer.layers.events.begin_batch.connect(
            self._pause_scene_graph_update
        )
        self.viewer.layers.events.end_batch.connect(
            self._resume_scene_graph_update
        )
        self.viewer.grid.events.stride.connect(self._update_scenegraph)
        self.viewer.grid.events.shape.connect(self._update_scenegraph)
        self.viewer.grid.events.enabled.connect(self._update_scenegraph)
        self.viewer.grid.events.spacing.connect(self._update_scenegraph)
        self.viewer._overlays.events.added.connect(
            self._update_viewer_overlays
        )
        self.viewer._overlays.events.removed.connect(
            self._update_viewer_overlays
        )
        self.viewer._overlays.events.changed.connect(
            self._update_viewer_overlays
        )
        self.destroyed.connect(self._disconnect_events)

    @property
    def events(self):
        # This is backwards compatible with the old events system
        # https://github.com/napari/napari/issues/7054#issuecomment-2205548968
        return self._scene_canvas.events

    @property
    def destroyed(self) -> pyqtBoundSignal:
        return self._scene_canvas._backend.destroyed

    @property
    def native(self) -> CanvasBackendDesktop:
        """Returns the native widget of the Vispy SceneCanvas."""
        return self._scene_canvas.native

    @property
    def screen_changed(self) -> Callable:
        """Bound method returning signal indicating whether the window screen has changed."""
        return self._scene_canvas._backend.screen_changed

    @property
    def background_color_override(self) -> str | npt.ArrayLike | None:
        """Background color of VispyCanvas.

        When not None, color is shown instead of VispyCanvas.bgcolor.
        """
        return self._background_color_override

    @background_color_override.setter
    def background_color_override(
        self, value: str | npt.ArrayLike | None
    ) -> None:
        self._background_color_override = value
        self.bgcolor = value or self._last_theme_color

    def _on_theme_change(self, event: Event) -> None:
        self._set_theme_change(event.value)

    def _set_theme_change(self, theme: str) -> None:
        from napari.utils.theme import get_theme

        # Note 1. store last requested theme color, in case we need to reuse it
        # when clearing the background_color_override, without needing to
        # keep track of the viewer.
        # Note 2. the reason for using the `as_hex` here is to avoid
        # `UserWarning` which is emitted when RGB values are above 1
        self._last_theme_color = transform_color(
            get_theme(theme).canvas.as_hex()
        )[0]
        self.bgcolor = self._last_theme_color

    def _disconnect_events(self) -> None:
        disconnect_events(self.viewer.events, self)
        disconnect_events(self.viewer._overlays.events, self)
        disconnect_events(self.viewer.camera.events, self)
        disconnect_events(self.viewer.layers.events, self)
        disconnect_events(self.viewer.camera.events, self)
        disconnect_events(self.viewer.cursor.events, self)
        disconnect_events(self._scene_canvas.events, self)

    @property
    def bgcolor(self) -> str:
        """Background color of the vispy scene canvas as a hex string. The setter expects str
        (any in vispy.color.get_color_names) or hex starting with # or a tuple | np.array ({3,4},)
        with values between 0 and 1."""
        return self._scene_canvas.bgcolor.hex

    @bgcolor.setter
    def bgcolor(self, value: str | npt.ArrayLike) -> None:
        self._scene_canvas.bgcolor = self._background_color_override or value

    @property
    def central_widget(self) -> Widget:
        """Overrides SceneCanvas.central_widget to make border_width=0"""
        if self._scene_canvas._central_widget is None:
            self._scene_canvas._central_widget = Widget(
                size=self.size,
                parent=self._scene_canvas.scene,
                border_width=0,
            )
        return self._scene_canvas._central_widget

    @property
    def size(self) -> tuple[int, int]:
        """Return canvas size as tuple (height, width) or accepts size as tuple (height, width)
        and sets Vispy SceneCanvas size as (width, height)."""
        return self._scene_canvas.size[::-1]

    @size.setter
    def size(self, size: tuple[int, int]):
        self._scene_canvas.size = size[::-1]

    @property
    def cursor(self) -> QCursor:
        """Cursor associated with native widget"""
        return self.native.cursor()

    @cursor.setter
    def cursor(self, q_cursor: QCursor | Qt.CursorShape):
        """Setting the cursor of the native widget"""
        self.native.setCursor(q_cursor)

    def _on_cursor(self) -> None:
        """Create a QCursor based on the napari cursor settings and set in Vispy."""
        cursor = self.viewer.cursor.style
        brush_overlay = self.viewer._brush_circle_overlay
        brush_overlay.visible = False

        if cursor in {'square', 'circle', 'circle_frozen'}:
            # Scale size by zoom if needed
            size = self.viewer.cursor.size
            if self.viewer.cursor.scaled:
                size *= self.viewer.camera.zoom

            size = int(size)

            # make sure the square fits within the current canvas
            if (
                size < 8 or size > (min(*self.size) - 4)
            ) and cursor != 'circle_frozen':
                self.cursor = QtCursorVisual['cross'].value
            elif cursor.startswith('circle'):
                brush_overlay.size = size
                if cursor == 'circle_frozen':
                    self.cursor = QtCursorVisual['standard'].value
                    brush_overlay.position_is_frozen = True
                else:
                    self.cursor = QtCursorVisual.blank()
                    brush_overlay.position_is_frozen = False
                brush_overlay.visible = True
            else:
                self.cursor = QtCursorVisual.square(size)
        elif cursor == 'crosshair':
            self.cursor = QtCursorVisual.crosshair()
        else:
            self.cursor = QtCursorVisual[cursor].value

    def delete(self) -> None:
        """Schedules the native widget for deletion"""
        self.native.deleteLater()

    def _on_interactive(self) -> None:
        """Link interactive attributes of view and viewer."""
        # Is this should be changed or renamed?
        interactive = (
            self.viewer.camera.mouse_zoom or self.viewer.camera.mouse_pan
        )
        if self.viewer.grid.enabled:
            self.view.interactive = False
            self.grid.interactive = interactive
        else:
            self.view.interactive = interactive
            self.grid.interactive = False

    def _on_boxzoom(self, event):
        """Update zoom level."""
        box_size_canvas = np.abs(
            np.diff(self.viewer._zoom_box.position, axis=0)
        )
        box_center_world = np.mean(event.value, axis=0)
        ratio = np.min(self._current_viewbox_size / box_size_canvas)
        self.viewer.camera.zoom = self.viewer.camera.zoom * np.min(ratio)
        self.viewer.camera.center = box_center_world

    def _map_canvas2world(
        self,
        position: tuple[int, ...],
        view: ViewBox,
    ) -> tuple[float, float]:
        """Map position from canvas pixels into world coordinates.

        Parameters
        ----------
        position : list(int, int)
            Position in canvas (x, y).

        Returns
        -------
        coords : tuple of two floats
            Position in world coordinates, matches the total dimensionality
            of the viewer.
        """
        nd = self.viewer.dims.ndisplay

        transform = view.transform * view.scene.transform

        # cartesian to homogeneous coordinates
        mapped_position = transform.imap(list(position))
        if nd == 3:
            mapped_position = mapped_position[0:nd] / mapped_position[nd]
        else:
            mapped_position = mapped_position[0:nd]
        position_world_slice = np.array(mapped_position[::-1])
        # handle position for 3D views of 2D data
        nd_point = len(self.viewer.dims.point)
        if nd_point < nd:
            position_world_slice = position_world_slice[-nd_point:]

        position_world = list(self.viewer.dims.point)
        for i, d in enumerate(self.viewer.dims.displayed):
            position_world[d] = position_world_slice[i]

        return tuple(position_world)

    def _get_viewbox_at(self, position):
        """Get the viewbox and its grid coordinates from the mouse position."""
        if not self.viewer.grid.enabled:
            return self.view, (0, 0)

        for (coords, _), viewbox in zip(
            self.viewer.grid.iter_viewboxes(len(self.viewer.layers)),
            self.grid_views,
            strict=False,
        ):
            shifted_pos = position - viewbox.transform.translate[:2]
            if viewbox.inner_rect.contains(*shifted_pos):
                return viewbox, coords

        return None, None

    def _process_mouse_event(
        self, mouse_callbacks: Callable, event: MouseEvent
    ) -> None:
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
        mouse_callbacks : Callable
            Mouse callbacks function.
        event : vispy.app.canvas.MouseEvent
            The vispy mouse event that triggered this method.

        Returns
        -------
        None
        """
        if event.pos is None:
            return

        # ensure that events which began in a specific viewbox continue to be
        # calculated based on that viewbox's coordinates
        if event.press_event is not None:
            viewbox, grid_coords = self._get_viewbox_at(event.press_event.pos)
        else:
            viewbox, grid_coords = self._get_viewbox_at(event.pos)

        self.viewer.cursor.viewbox = grid_coords

        if viewbox is None:
            # this means we're in an empty viewbox, so do nothing
            event.handled = True
            return

        napari_event = NapariMouseEvent(
            event=event,
            view_direction=self._calculate_view_direction(event.pos),
            up_direction=self.viewer.camera.calculate_nd_up_direction(
                self.viewer.dims.ndim, self.viewer.dims.displayed
            ),
            camera_zoom=self.viewer.camera.zoom,
            position=self._map_canvas2world(event.pos, viewbox),
            dims_displayed=list(self.viewer.dims.displayed),
            dims_point=list(self.viewer.dims.point),
            viewbox=grid_coords,
        )

        # Update the cursor position
        self.viewer.cursor._view_direction = napari_event.view_direction
        self.viewer.cursor.position = napari_event.position

        # Put a read only wrapper on the event
        read_only_event = ReadOnlyWrapper(
            napari_event, exceptions=('handled',)
        )
        mouse_callbacks(self.viewer, read_only_event)

        layer = self.viewer.layers.selection.active
        if layer is not None:
            mouse_callbacks(layer, read_only_event)

        event.handled = napari_event.handled

    def _on_mouse_double_click(self, event: MouseEvent) -> None:
        """Called whenever a mouse double-click happen on the canvas

        Parameters
        ----------
        event : vispy.app.canvas.MouseEvent
            The vispy mouse event that triggered this method. The `event.type` will always be `mouse_double_click`

        Returns
        -------
        None

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

    def _on_mouse_move(self, event: MouseEvent) -> None:
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.

        Returns
        -------
        None
        """
        self._process_mouse_event(mouse_move_callbacks, event)

    def _on_mouse_press(self, event: MouseEvent) -> None:
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : vispy.app.canvas.MouseEvent
            The vispy mouse event that triggered this method.

        Returns
        -------
        None
        """
        self._process_mouse_event(mouse_press_callbacks, event)

    def _on_mouse_release(self, event: MouseEvent) -> None:
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : vispy.app.canvas.MouseEvent
            The vispy mouse event that triggered this method.

        Returns
        -------
        None
        """
        self._process_mouse_event(mouse_release_callbacks, event)

    def _on_mouse_wheel(self, event: MouseEvent) -> None:
        """Called whenever mouse wheel activated in canvas.

        Parameters
        ----------
        event : vispy.app.canvas.MouseEvent
            The vispy mouse event that triggered this method.

        Returns
        -------
        None
        """
        self._process_mouse_event(mouse_wheel_callbacks, event)

    @property
    def _viewbox_corners_in_world(self) -> npt.NDArray:
        """Location of the corners of canvas in world coordinates.

        Returns
        -------
        corners : np.ndarray
            Coordinates of top left and bottom right canvas pixel in the world.
        """
        if self.viewer.grid.enabled and self.grid_views:
            # they are all the same, just take the first one
            view = self.grid_views[0]
        else:
            view = self.view

        # Find corners of canvas in world coordinates
        top_left = self._map_canvas2world((0, 0), view)
        bottom_right = self._map_canvas2world(view.rect.size, view)
        return np.array([top_left, bottom_right])

    def on_draw(self, event: DrawEvent | None = None) -> None:
        """Called whenever the canvas is drawn.

        This is triggered from vispy whenever new data is sent to the canvas or
        the camera is moved and is connected in the `QtViewer`.

        Parameters
        ----------
        event : vispy.app.canvas.DrawEvent
            The draw event from the vispy canvas.

        Returns
        -------
        None
        """
        # this updates camera zooms and overlay positions if necessary
        # (usually after grid mode enable when viewboxes are still degenerate)
        if not np.allclose(
            self._last_viewbox_size, self._current_viewbox_size
        ):
            self._update_grid_spacing()
            self._last_viewbox_size = self._current_viewbox_size
            self._needs_overlay_position_update = True

        if self._needs_overlay_position_update:
            self._update_overlay_canvas_positions()
            self._needs_overlay_position_update = False

        # sync all cameras
        for camera in (self.camera, *self.grid_cameras):
            camera.on_draw(event)

        # The canvas corners in full world coordinates (i.e. across all layers).
        viewbox_corners_world = self._viewbox_corners_in_world
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
                displayed_axes = list(self.viewer.dims.displayed[-nd:])
            layer._update_draw(
                scale_factor=1 / self.viewer.camera.zoom,
                corner_pixels_displayed=viewbox_corners_world[
                    :, displayed_axes
                ],
                shape_threshold=self._current_viewbox_size[::-1],
            )

    def on_resize(self, event: ResizeEvent) -> None:
        """Called whenever canvas is resized.

        Parameters
        ----------
        event : vispy.app.canvas.ResizeEvent
            The vispy event that triggered this method.

        Returns
        -------
        None
        """
        self.viewer._canvas_size = self.size

    def add_layer_visual_mapping(
        self, napari_layer: Layer, vispy_layer: VispyBaseLayer
    ) -> None:
        """Maps a napari layer to its corresponding vispy layer and sets the parent scene of the vispy layer.

        Parameters
        ----------
        napari_layer :
            Any napari layer, the layer type is the same as the vispy layer.
        vispy_layer :
            Any vispy layer, the layer type is the same as the napari layer.

        Returns
        -------
        None
        """
        self.layer_to_visual[napari_layer] = vispy_layer
        self._layer_overlay_to_visual[napari_layer] = {}

        napari_layer.events.visible.connect(self._reorder_layers)
        overlay_callback = partial(self._update_layer_overlays, napari_layer)
        napari_layer.events.visible.connect(overlay_callback)
        napari_layer._overlays.events.added.connect(overlay_callback)
        napari_layer._overlays.events.removed.connect(overlay_callback)
        napari_layer._overlays.events.changed.connect(overlay_callback)
        self._overlay_callbacks[napari_layer] = overlay_callback
        self.viewer.camera.events.angles.connect(vispy_layer._on_camera_move)

        # we need to trigger _on_matrix_change once after adding the overlays so that
        # all children nodes are assigned the correct transforms
        vispy_layer._on_matrix_change()
        self._update_scenegraph()

    def _remove_layer(self, event: Event) -> None:
        """Upon receiving event closes the Vispy visual, deletes it and reorders the still existing layers.

        Parameters
        ----------
        event : napari.utils.events.event.Event
            The event causing a particular layer to be removed

        Returns
        -------
        None
        """
        layer = event.value
        disconnect_events(layer.events, self)
        disconnect_events(layer.events, self._overlay_callbacks[layer])
        disconnect_events(
            layer._overlays.events, self._overlay_callbacks[layer]
        )
        del self._overlay_callbacks[layer]

        vispy_layer = self.layer_to_visual.pop(layer)
        disconnect_events(self.viewer.camera.events, vispy_layer)
        vispy_layer.close()
        del vispy_layer

        self._update_layer_overlays(layer)
        del self._layer_overlay_to_visual[layer]
        if self._pause_scene_graph:
            return
        self._clean_and_update_scenegraph()

    def _clean_and_update_scenegraph(self):
        # Critical two-step fix for Windows OpenGL access violation bug
        # This prevents the race condition where scenegraph updates occur while
        # GPU resources from the removed layer are still being processed/deleted.

        # Step 1: Force immediate garbage collection of OpenGL resources
        # When vispy_layer.close() is called above, it marks OpenGL objects (textures,
        # buffers, etc.) for deletion, but Python's garbage collector may not run
        # immediately. On some Windows OpenGL drivers (especially NVIDIA cards),
        # this can leave "dangling" OpenGL resource references.
        gc.collect()

        # Step 2: Synchronize the OpenGL command queue
        # Layer removal involves deleting GPU textures, buffers, and shader programs/
        # The subsequent _update_scenegraph() call rebuilds the scene graph with new GPU resources.
        # If the GPU is still processing deletion commands from the removed layer, it can lead to
        # memory access violations when the scene graph is updated with new resources.
        # finish() ensures complete GPU synchronization before proceeding
        self._scene_canvas.context.finish()

        self._update_scenegraph()

    def _reorder_layers(self) -> None:
        """When the list is reordered, propagate changes to draw order."""
        if self.viewer.grid.enabled:
            for _, layer_indices in self.viewer.grid.iter_viewboxes(
                len(self.viewer.layers)
            ):
                if not layer_indices:
                    continue
                layers = [self.viewer.layers[idx] for idx in layer_indices]
                self._reorder_layers_in_the_same_view(layers)
        else:
            self._reorder_layers_in_the_same_view(self.viewer.layers)

    def _reorder_layers_in_the_same_view(self, layers):
        first_visible_found = False

        for i, layer in enumerate(layers):
            vispy_layer = self.layer_to_visual[layer]
            vispy_layer.order = i

            # the bottommost visible layer needs special treatment for blending
            if layer.visible and not first_visible_found:
                vispy_layer.first_visible = True
                first_visible_found = True
            else:
                vispy_layer.first_visible = False
            vispy_layer._on_blending_change()

        self._defer_overlay_position_update()

        self._scene_canvas._draw_order.clear()
        self._scene_canvas.update()

    def _defer_overlay_position_update(self):
        self._needs_overlay_position_update = True

    def _connect_canvas_overlay_events(self, overlay: Overlay) -> None:
        overlay.events.position.connect(self._defer_overlay_position_update)
        overlay.events.visible.connect(self._defer_overlay_position_update)

    def _disconnect_canvas_overlay_events(self, overlay: Overlay) -> None:
        overlay.events.position.disconnect(self._defer_overlay_position_update)
        overlay.events.visible.disconnect(self._defer_overlay_position_update)

    def _update_viewer_overlays(self):
        """Update the viewer overlay visuals.

        Also ensures that overlays are properly assigned parents depending on
        their class (canvas vs scene overlays).
        """
        # delete outdated overlays
        for overlay in set(self._overlay_to_visual) - set(
            self.viewer._overlays.values()
        ):
            if isinstance(overlay, CanvasOverlay):
                self._disconnect_canvas_overlay_events(overlay)
            vispy_overlays = self._overlay_to_visual.pop(overlay)
            for vispy_overlay in vispy_overlays:
                vispy_overlay.close()

        for overlay in self.viewer._overlays.values():
            vispy_overlays = self._overlay_to_visual.setdefault(overlay, [])

            # connect position callbacks
            if isinstance(overlay, CanvasOverlay):
                self._connect_canvas_overlay_events(overlay)
                overlay.events.gridded.connect(self._update_viewer_overlays)

            if self.viewer.grid.enabled and getattr(overlay, 'gridded', True):
                views = self.grid_views
            else:
                views = [self.view]

            for view, vispy_overlay in zip_longest(views, vispy_overlays):
                if view is None:
                    # works backwards but ends up removing the correct amount
                    # when the number of views decreased since last time
                    vispy_overlays.pop().close()
                    continue

                parent = (
                    view if isinstance(overlay, CanvasOverlay) else view.scene
                )

                if vispy_overlay is None:
                    vispy_overlay = create_vispy_overlay(
                        overlay=overlay, viewer=self.viewer, parent=parent
                    )
                    vispy_overlays.append(vispy_overlay)

                    if isinstance(overlay, CanvasOverlay):
                        vispy_overlay.canvas_position_callback = (
                            self._defer_overlay_position_update
                        )

                else:
                    vispy_overlay.node.parent = parent

        self._defer_overlay_position_update()

    def _update_layer_overlays(self, layer: Layer) -> None:
        """Update the overlay visuals for each layer in the canvas.

        Also ensures that overlays are properly assigned parents depending on
        their class (canvas vs scene overlays).
        """
        overlay_to_visual = self._layer_overlay_to_visual.setdefault(layer, {})

        # delete outdated overlays
        if layer not in self.viewer.layers:
            to_remove = set(layer._overlays.values())
        else:
            to_remove = set(overlay_to_visual.keys()) - set(
                layer._overlays.values()
            )

        for overlay in to_remove:
            if isinstance(overlay, CanvasOverlay):
                self._disconnect_canvas_overlay_events(overlay)
            if vispy_overlay := overlay_to_visual.pop(overlay, None):
                vispy_overlay.close()

        if layer not in self.viewer.layers:
            # we're just removing all the overlays of this layer, so we're done here
            return

        for overlay in layer._overlays.values():
            # only create overlays when they are visible. If not, we connect the visible
            # event of this overlay to this method until it's finally visible
            if not overlay.visible:
                overlay.events.visible.connect(self._overlay_callbacks[layer])
                continue
            overlay.events.visible.disconnect(self._overlay_callbacks[layer])

            vispy_overlay = overlay_to_visual.get(overlay, None)

            if isinstance(overlay, CanvasOverlay):
                self._connect_canvas_overlay_events(overlay)

                if self.viewer.grid.enabled:
                    row, col = self.viewer.grid.position(
                        self.viewer.layers.index(layer),
                        len(self.viewer.layers),
                    )
                    parent = self.grid[row, col]
                else:
                    parent = self.view
            else:
                parent = self.layer_to_visual[layer].node

            if vispy_overlay is None:
                vispy_overlay = create_vispy_overlay(
                    overlay=overlay, layer=layer, parent=parent
                )
                overlay_to_visual[overlay] = vispy_overlay
                if isinstance(overlay, CanvasOverlay):
                    vispy_overlay.canvas_position_callback = (
                        self._defer_overlay_position_update
                    )
            else:
                vispy_overlay.node.parent = parent

        self._defer_overlay_position_update()

    def _get_ordered_visible_canvas_overlays(
        self,
    ) -> Iterator[tuple[CanvasOverlay, VispyBaseOverlay, Node | None]]:
        """
        Iterator over visible canvas overlays by grid viewbox, in tiling order.

        Returns a tuple containing the overlay model, its matching visual, and
        the index of the view in the grid where the overlay should be displayed.
        The view `None` is special cased to refer to the base, non-gridded view.

        Note that some canvas overlays do no use CanvasPosition, but are instead
        free-floating (such as the cursor overlay), so those are skipped
        """

        def is_visible_tileable(overlay):
            return (
                overlay.visible
                and isinstance(overlay, CanvasOverlay)
                and overlay.position in list(CanvasPosition)
            )

        def is_gridded(overlay):
            return overlay.gridded and self.viewer.grid.enabled

        # first the base view: non-gridded viewer overlays which appear
        # "on top of" the main canvas
        for overlay, vispy_overlays in self._overlay_to_visual.items():
            if is_visible_tileable(overlay) and not is_gridded(overlay):
                yield overlay, vispy_overlays[0], None

        # then gridded viewer overlays and layer overlays, by viewbox, in order
        for viewbox_idx, (_, layer_indices) in enumerate(
            self.viewer.grid.iter_viewboxes(len(self.viewer.layers))
        ):
            if not layer_indices:
                # last empty boxes of the grid
                break

            # if grid is disabled, this loop runs once and we put everything
            # in the base (None) viewbox
            view = viewbox_idx if self.viewer.grid.enabled else None

            for overlay, vispy_overlays in self._overlay_to_visual.items():
                if is_visible_tileable(overlay) and is_gridded(overlay):
                    yield overlay, vispy_overlays[viewbox_idx], view

            # layer overlays are always "gridded"
            # (they always appear in the same viewbox as the layer itself)
            for layer_idx in layer_indices:
                layer = self.viewer.layers[layer_idx]
                for (
                    overlay,
                    vispy_overlay,
                ) in self._layer_overlay_to_visual.get(layer, {}).items():
                    if layer.visible and is_visible_tileable(overlay):
                        yield overlay, vispy_overlay, view

    def _update_overlay_canvas_positions(self, event=None):
        # TODO: make settable
        x_padding = y_padding = 10.0
        x_offset_total = {}
        y_offset_total = {}
        for (
            overlay,
            vispy_overlay,
            view,
        ) in self._get_ordered_visible_canvas_overlays():
            # TODO: vertical vs horizontal tiling should be settable!
            x_offset_total.setdefault(
                view, dict.fromkeys(CanvasPosition, x_padding)
            )
            y_offset_total.setdefault(
                view, dict.fromkeys(CanvasPosition, y_padding)
            )

            x_offset = x_offset_total[view][overlay.position]
            y_offset = y_offset_total[view][overlay.position]

            # add offset to the following overlays based on tiling direction
            # these are currently hardcoded, so we just tile horizontally or
            # vertically depending on which corner we're on
            if overlay.position in ('top_right', 'bottom_left'):
                x_offset_total[view][overlay.position] += (
                    vispy_overlay.x_size + x_padding
                )
            else:
                y_offset_total[view][overlay.position] += (
                    vispy_overlay.y_size + y_padding
                )

            # position the overlay in the canvas
            # if the overlay is in a grid viewbox, use the viewbox size
            if view is None:
                y_max, x_max = self.size
            else:
                x_max, y_max = self._current_viewbox_size
            position = overlay.position

            x = y = 0
            if 'top' in position:
                y = y_offset
            elif 'bottom' in position:
                y = y_max - vispy_overlay.y_size - y_offset

            if 'left' in position:
                x = x_offset
            elif 'right' in position:
                x = x_max - vispy_overlay.x_size - x_offset
            elif 'center' in position:
                x = x_max / 2 - vispy_overlay.x_size / 2

            vispy_overlay.node.transform.translate = [x, y, 0, 0]

        self._needs_overlay_position_update = False

    def _calculate_view_direction(
        self, event_pos: tuple[float, float]
    ) -> npt.NDArray[np.float64] | None:
        """calculate view direction by ray shot from the camera"""
        # this method is only implemented for 3 dimension
        if self.viewer.dims.ndisplay == 2:
            return None

        if self.viewer.dims.ndim == 2:
            return self.viewer.camera.calculate_nd_view_direction(
                self.viewer.dims.ndim, self.viewer.dims.displayed
            )
        x, y = event_pos
        w, h = self.size
        nd = self.viewer.dims.ndisplay

        view = self._get_viewbox_at(event_pos)[0] or self.view
        # combine the viewbox transform wit the scene transform
        # so each viewbox in grid mode maps back to the main scene
        transform = view.transform * view.scene.transform

        # map click pos to scene coordinates
        click_scene = transform.imap([x, y, 0, 1])
        # canvas center at infinite far z- (eye position in canvas coordinates)
        eye_canvas = [w / 2, h / 2, -1e10, 1]
        # map eye pos to scene coordinates
        eye_scene = transform.imap(eye_canvas)
        # homogeneous coordinate to cartesian
        click_scene = click_scene[0:nd] / click_scene[nd]
        # homogeneous coordinate to cartesian
        eye_scene = eye_scene[0:nd] / eye_scene[nd]

        # calculate direction of the ray
        d = click_scene - eye_scene
        d = d[0:nd]
        d = d / np.linalg.norm(d)
        # xyz to zyx
        d: list[float] = list(d[::-1])
        # convert to nd view direction
        view_direction_nd = np.zeros(self.viewer.dims.ndim, dtype=np.float64)
        view_direction_nd[list(self.viewer.dims.displayed)] = d
        return view_direction_nd

    def screenshot(self) -> QImage:
        """Return a QImage based on what is shown in the viewer."""
        # ensure on_draw is run to bring everything up to date
        # needed for some Ubuntu py3.10 pyqt5 tests, but likely inconsistent behavior for other OS.
        # See: https://github.com/napari/napari/pull/7870#issuecomment-2997167180
        self.on_draw(None)
        return self.native.grabFramebuffer()

    def enable_dims_play(self, *args) -> None:
        """Enable playing of animation. False if awaiting a draw event"""
        self.viewer.dims._play_ready = True

    def _init_or_update_grid(self) -> None:
        # grid are really not designed to be reset, so we have to replace it
        # when necessary (every time the grid shape changes)
        if self.grid.grid_size == self.viewer.grid.actual_shape(
            len(self.viewer.layers)
        ):
            return

        for camera in self.grid_cameras:
            self._scene_canvas.events.draw.disconnect(camera.on_draw)
            camera._2D_camera.parent = None
            camera._3D_camera.parent = None
        self.grid_cameras.clear()
        self.grid_views.clear()
        self.grid.parent = None

        self.grid = self.central_widget.add_grid(border_width=0)

        for (row, col), _ in self.viewer.grid.iter_viewboxes(
            len(self.viewer.layers)
        ):
            view = self.grid[row, col]
            # any border_color != None will add a padding of +1
            # see https://github.com/vispy/vispy/issues/1492
            view.border_width = 0
            view.border_color = None

            camera = VispyCamera(view, self.viewer.camera, self.viewer.dims)
            self.grid_views.append(view)
            self.grid_cameras.append(camera)

    def _update_scenegraph(self, event=None):
        if self._pause_scene_graph:
            return
        with self._scene_canvas.events.draw.blocker():
            if self.viewer.grid.enabled:
                self._init_or_update_grid()
                self._setup_layer_views_in_grid()
                self._update_grid_spacing()
            else:
                self._setup_single_view()

            self._reorder_layers()
            self._update_viewer_overlays()
            for layer in self.viewer.layers:
                self._update_layer_overlays(layer)
            self._on_interactive()
        self.on_draw(None)

    def _setup_single_view(self):
        for vispy_layer in self.layer_to_visual.values():
            vispy_layer.node.parent = self.view.scene

    def _setup_layer_views_in_grid(self):
        for (row, col), layer_indices in self.viewer.grid.iter_viewboxes(
            len(self.viewer.layers)
        ):
            view = self.grid[row, col]

            for idx in layer_indices:
                napari_layer = self.viewer.layers[idx]
                vispy_layer = self.layer_to_visual[napari_layer]
                vispy_layer.node.parent = view.scene

    @property
    def _current_viewbox_size(self):
        """Get the actual size of the viewboxes in pixels.

        If grid is not enabled, this returns the size of the canvas.
        If grid is enabled, this returns the size of the viewboxes; note that
        these can be degenerate if `on_draw` hasn't been called yet!

        Returns
        -------
        tuple[int, int]
            The size of the viewbox(es) in pixels (width, height)
        """
        if self.viewer.grid.enabled and self.grid_views:
            return self.grid_views[0].rect.size

        return self.view.rect.size

    def _update_grid_spacing(self):
        """Update the grid spacing with a validated spacing value.

        This method computes the raw spacing based on the current canvas size
        and validates it against the maximum safe spacing. If the raw spacing
        exceeds the maximum safe spacing, it is reduced to the maximum safe value
        and a warning is issued to the user.
        """
        # TODO: this should be all handled on the grid model ideally, using validators
        raw_spacing = self.viewer.grid._compute_canvas_spacing_raw(
            self._scene_canvas.size, len(self.viewer.layers)
        )
        safe_spacing = self.viewer.grid._compute_canvas_spacing(
            self._scene_canvas.size, len(self.viewer.layers)
        )

        if raw_spacing > safe_spacing:
            warnings.warn(
                trans._(
                    'Grid spacing of {raw_spacing:.1f} pixels is too large and has been '
                    'reduced to {safe_spacing:.1f} pixels to prevent viewboxes from '
                    'becoming too small. Consider using a smaller spacing value or '
                    'increasing the canvas size.',
                    deferred=True,
                    raw_spacing=raw_spacing,
                    safe_spacing=safe_spacing,
                ),
                UserWarning,
                stacklevel=2,
            )
            # this shouldn't cause an infinite loop cause now the spacing is fixed!
            self.viewer.grid.spacing = safe_spacing

        self.grid.spacing = safe_spacing

    def _pause_scene_graph_update(self):
        self._pause_scene_graph = True

    def _resume_scene_graph_update(self):
        self._pause_scene_graph = False
        self._clean_and_update_scenegraph()
