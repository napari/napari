from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import QCoreApplication, QObject, Qt
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QFileDialog, QSplitter, QVBoxLayout, QWidget

from ..components.camera import Camera
from ..components.layerlist import LayerList
from ..utils import config, perf
from ..utils.action_manager import action_manager
from ..utils.history import (
    get_open_history,
    get_save_history,
    update_open_history,
    update_save_history,
)
from ..utils.interactions import (
    ReadOnlyWrapper,
    mouse_double_click_callbacks,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)
from ..utils.io import imsave
from ..utils.key_bindings import KeymapHandler
from ..utils.misc import in_ipython
from ..utils.theme import get_theme
from ..utils.translations import trans
from .containers import QtLayerList
from .dialogs.screenshot_dialog import ScreenshotDialog
from .perf.qt_performance import QtPerformance
from .utils import QImg2array, circle_pixmap, square_pixmap
from .widgets.qt_dims import QtDims
from .widgets.qt_viewer_buttons import QtLayerButtons, QtViewerButtons
from .widgets.qt_viewer_dock_widget import QtViewerDockWidget
from .widgets.qt_welcome import QtWidgetOverlay

from .._vispy import (  # isort:skip
    VispyAxesVisual,
    VispyCamera,
    VispyCanvas,
    VispyScaleBarVisual,
    VispyTextVisual,
    create_vispy_visual,
)


if TYPE_CHECKING:
    from ..viewer import Viewer

from ..settings import get_settings
from ..utils.io import imsave_extensions


class QtViewer(QSplitter):
    """Qt view for the napari Viewer model.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    show_welcome_screen : bool, optional
        Flag to show a welcome message when no layers are present in the
        canvas. Default is `False`.

    Attributes
    ----------
    canvas : vispy.scene.SceneCanvas
        Canvas for rendering the current view.
    console : QtConsole
        IPython console terminal integrated into the napari GUI.
    controls : QtLayerControlsContainer
        Qt view for GUI controls.
    dims : napari.qt_dims.QtDims
        Dimension sliders; Qt View for Dims model.
    dockConsole : QtViewerDockWidget
        QWidget wrapped in a QDockWidget with forwarded viewer events.
    aboutKeybindings : QtAboutKeybindings
        Key bindings for the 'About' Qt dialog.
    dockLayerControls : QtViewerDockWidget
        QWidget wrapped in a QDockWidget with forwarded viewer events.
    dockLayerList : QtViewerDockWidget
        QWidget wrapped in a QDockWidget with forwarded viewer events.
    layerButtons : QtLayerButtons
        Button controls for napari layers.
    layers : QtLayerList
        Qt view for LayerList controls.
    layer_to_visual : dict
        Dictionary mapping napari layers with their corresponding vispy_layers.
    view : vispy scene widget
        View displayed by vispy canvas. Adds a vispy ViewBox as a child widget.
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    viewerButtons : QtViewerButtons
        Button controls for the napari viewer.
    """

    def __init__(self, viewer: Viewer, show_welcome_screen: bool = False):
        # Avoid circular import.
        from .layer_controls import QtLayerControlsContainer

        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._show_welcome_screen = show_welcome_screen

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer
        self.dims = QtDims(self.viewer.dims)
        self.controls = QtLayerControlsContainer(self.viewer)
        self.layers = QtLayerList(self.viewer.layers)
        self.layerButtons = QtLayerButtons(self.viewer)
        self.viewerButtons = QtViewerButtons(self.viewer)
        self._key_map_handler = KeymapHandler()
        self._key_map_handler.keymap_providers = [self.viewer]
        self._console = None

        layerList = QWidget()
        layerList.setObjectName('layerList')
        layerListLayout = QVBoxLayout()
        layerListLayout.addWidget(self.layerButtons)
        layerListLayout.addWidget(self.layers)
        layerListLayout.addWidget(self.viewerButtons)
        layerListLayout.setContentsMargins(8, 4, 8, 6)
        layerList.setLayout(layerListLayout)

        self.dockLayerList = QtViewerDockWidget(
            self,
            layerList,
            name=trans._('layer list'),
            area='left',
            allowed_areas=['left', 'right'],
            object_name='layer list',
        )
        self.dockLayerControls = QtViewerDockWidget(
            self,
            self.controls,
            name=trans._('layer controls'),
            area='left',
            allowed_areas=['left', 'right'],
            object_name='layer controls',
        )
        self.dockConsole = QtViewerDockWidget(
            self,
            QWidget(),
            name=trans._('console'),
            area='bottom',
            allowed_areas=['top', 'bottom'],
            object_name='console',
        )
        self.dockConsole.setVisible(False)
        # because the console is loaded lazily in the @getter, this line just
        # gets (or creates) the console when the dock console is made visible.
        self.dockConsole.visibilityChanged.connect(
            lambda visible: self.console if visible else None
        )
        self.dockLayerControls.visibilityChanged.connect(self._constrain_width)
        self.dockLayerList.setMaximumWidth(258)
        self.dockLayerList.setMinimumWidth(258)

        # Only created if using perfmon.
        self.dockPerformance = self._create_performance_dock_widget()

        # This dictionary holds the corresponding vispy visual for each layer
        self.layer_to_visual = {}
        action_manager.register_action(
            "napari:toggle_console_visibility",
            self.toggle_console_visibility,
            trans._("Show/Hide IPython console"),
            self.viewer,
        )
        action_manager.bind_button(
            'napari:toggle_console_visibility',
            self.viewerButtons.consoleButton,
        )

        self._create_canvas()

        # Stacked widget to provide a welcome page
        self._canvas_overlay = QtWidgetOverlay(self, self.canvas.native)
        self._canvas_overlay.set_welcome_visible(show_welcome_screen)
        self._canvas_overlay.sig_dropped.connect(self.dropEvent)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 22, 10, 2)
        main_layout.addWidget(self._canvas_overlay)
        main_layout.addWidget(self.dims)
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)

        self.setOrientation(Qt.Vertical)
        self.addWidget(main_widget)

        self._cursors = {
            'cross': Qt.CrossCursor,
            'forbidden': Qt.ForbiddenCursor,
            'pointing': Qt.PointingHandCursor,
            'standard': QCursor(),
        }

        self._on_active_change()
        self.viewer.layers.events.inserted.connect(self._update_welcome_screen)
        self.viewer.layers.events.removed.connect(self._update_welcome_screen)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_change
        )
        self.viewer.camera.events.interactive.connect(self._on_interactive)
        self.viewer.cursor.events.style.connect(self._on_cursor)
        self.viewer.cursor.events.size.connect(self._on_cursor)
        self.viewer.layers.events.reordered.connect(self._reorder_layers)
        self.viewer.layers.events.inserted.connect(self._on_add_layer_change)
        self.viewer.layers.events.removed.connect(self._remove_layer)

        self.setAcceptDrops(True)

        for layer in self.viewer.layers:
            self._add_layer(layer)

        self.view = self.canvas.central_widget.add_view()
        self.camera = VispyCamera(
            self.view, self.viewer.camera, self.viewer.dims
        )
        self.canvas.connect(self.camera.on_draw)

        # Add axes, scale bar
        self._add_visuals()

        # Create the experimental QtPool for octree and/or monitor.
        self._qt_poll = _create_qt_poll(self, self.viewer.camera)

        # Create the experimental RemoteManager for the monitor.
        self._remote_manager = _create_remote_manager(
            self.viewer.layers, self._qt_poll
        )

        # moved from the old layerlist... still feels misplaced.
        # can you help me move this elsewhere?
        if config.async_loading:
            from .experimental.qt_chunk_receiver import QtChunkReceiver

            # The QtChunkReceiver object allows the ChunkLoader to pass newly
            # loaded chunks to the layers that requested them.
            self.chunk_receiver = QtChunkReceiver(self.layers)
        else:
            self.chunk_receiver = None

        # bind shortcuts stored in settings last.
        self._bind_shortcuts()

    def _bind_shortcuts(self):
        """Bind shortcuts stored in SETTINGS to actions."""
        for action, shortcuts in get_settings().shortcuts.shortcuts.items():
            action_manager.unbind_shortcut(action)
            for shortcut in shortcuts:
                action_manager.bind_shortcut(action, shortcut)

    def _create_canvas(self) -> None:
        """Create the canvas and hook up events."""
        self.canvas = VispyCanvas(
            keys=None,
            vsync=True,
            parent=self,
            size=self.viewer._canvas_size[::-1],
        )
        self.canvas.events.draw.connect(self.dims.enable_play)

        self.canvas.connect(self.on_mouse_double_click)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self._key_map_handler.on_key_press)
        self.canvas.connect(self._key_map_handler.on_key_release)
        self.canvas.connect(self.on_mouse_wheel)
        self.canvas.connect(self.on_draw)
        self.canvas.connect(self.on_resize)
        self.canvas.bgcolor = get_theme(self.viewer.theme)['canvas']
        theme = self.viewer.events.theme

        on_theme_change = self.canvas._on_theme_change
        theme.connect(on_theme_change)

        def disconnect():
            # strange EventEmitter has no attribute _callbacks errors sometimes
            # maybe some sort of cleanup race condition?
            with suppress(AttributeError):
                theme.disconnect(on_theme_change)

        self.canvas.destroyed.connect(disconnect)

    def _add_visuals(self) -> None:
        """Add visuals for axes, scale bar, and welcome text."""

        self.axes = VispyAxesVisual(
            self.viewer,
            parent=self.view.scene,
            order=1e6,
        )
        self.scale_bar = VispyScaleBarVisual(
            self.viewer,
            parent=self.view,
            order=1e6 + 1,
        )
        self.canvas.events.resize.connect(self.scale_bar._on_position_change)
        self.text_overlay = VispyTextVisual(
            self.viewer,
            parent=self.view,
            order=1e6 + 2,
        )
        self.canvas.events.resize.connect(
            self.text_overlay._on_position_change
        )

    def _create_performance_dock_widget(self):
        """Create the dock widget that shows performance metrics."""
        if perf.USE_PERFMON:
            return QtViewerDockWidget(
                self,
                QtPerformance(),
                name=trans._('performance'),
                area='bottom',
            )
        return None

    @property
    def console(self):
        """QtConsole: iPython console terminal integrated into the napari GUI."""
        if self._console is None:
            try:
                from napari_console import QtConsole

                import napari

                self.console = QtConsole(self.viewer)
                self.console.push(
                    {'napari': napari, 'action_manager': action_manager}
                )
            except ImportError:
                warnings.warn(
                    trans._(
                        'napari-console not found. It can be installed with'
                        ' "pip install napari_console"'
                    )
                )
                self._console = None
        return self._console

    @console.setter
    def console(self, console):
        self._console = console
        if console is not None:
            self.dockConsole.setWidget(console)
            console.setParent(self.dockConsole)

    def _constrain_width(self, event):
        """Allow the layer controls to be wider, only if floated.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if self.dockLayerControls.isFloating():
            self.controls.setMaximumWidth(700)
        else:
            self.controls.setMaximumWidth(220)

    def _on_active_change(self, event=None):
        """When active layer changes change keymap handler.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self._key_map_handler.keymap_providers = (
            [self.viewer]
            if self.viewer.layers.selection.active is None
            else [self.viewer.layers.selection.active, self.viewer]
        )

    def _on_add_layer_change(self, event):
        """When a layer is added, set its parent and order.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer = event.value
        self._add_layer(layer)

    def _add_layer(self, layer):
        """When a layer is added, set its parent and order.

        Parameters
        ----------
        layer : napari.layers.Layer
            Layer to be added.
        """
        vispy_layer = create_vispy_visual(layer)

        # QtPoll is experimental.
        if self._qt_poll is not None:
            # QtPoll will call VipyBaseImage._on_poll() when the camera
            # moves or the timer goes off.
            self._qt_poll.events.poll.connect(vispy_layer._on_poll)

            # In the other direction, some visuals need to tell QtPoll to
            # start polling. When they receive new data they need to be
            # polled to load it, even if the camera is not moving.
            if vispy_layer.events is not None:
                vispy_layer.events.loaded.connect(self._qt_poll.wake_up)

        vispy_layer.node.parent = self.view.scene
        vispy_layer.order = len(self.viewer.layers) - 1
        self.layer_to_visual[layer] = vispy_layer

    def _remove_layer(self, event):
        """When a layer is removed, remove its parent.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        layer = event.value
        vispy_layer = self.layer_to_visual[layer]
        vispy_layer.close()
        del vispy_layer
        self._reorder_layers(None)

    def _reorder_layers(self, event):
        """When the list is reordered, propagate changes to draw order.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        for i, layer in enumerate(self.viewer.layers):
            vispy_layer = self.layer_to_visual[layer]
            vispy_layer.order = i
        self.canvas._draw_order.clear()
        self.canvas.update()

    def _save_layers_dialog(self, selected=False):
        """Save layers (all or selected) to disk, using ``LayerList.save()``.

        Parameters
        ----------
        selected : bool
            If True, only layers that are selected in the viewer will be saved.
            By default, all layers are saved.
        """
        msg = ''
        if not len(self.viewer.layers):
            msg = trans._("There are no layers in the viewer to save")
        elif selected and not len(self.viewer.layers.selection):
            msg = trans._(
                'Please select one or more layers to save,'
                '\nor use "Save all layers..."'
            )
        if msg:
            raise OSError(trans._("Nothing to save"))

        # prepare list of extensions for drop down menu.
        if selected and len(self.viewer.layers.selection) == 1:
            selected_layer = list(self.viewer.layers.selection)[0]
            # single selected layer.
            if selected_layer._type_string == 'image':

                ext = imsave_extensions()

                ext_list = []
                for val in ext:
                    ext_list.append("*" + val)

                ext_str = ';;'.join(ext_list)

                ext_str = trans._(
                    "All Files (*);; Image file types:;;{ext_str}",
                    ext_str=ext_str,
                )

            elif selected_layer._type_string == 'points':

                ext_str = trans._("All Files (*);; *.csv;;")

            else:
                # layer other than image or points
                ext_str = trans._("All Files (*);;")

        else:
            # multiple layers.
            ext_str = trans._("All Files (*);;")

        msg = trans._("selected") if selected else trans._("all")
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)

        filename, _ = dlg.getSaveFileName(
            parent=self,
            caption=trans._('Save {msg} layers', msg=msg),
            directory=hist[0],  # home dir by default,
            filter=ext_str,
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if filename:
            with warnings.catch_warnings(record=True) as wa:
                saved = self.viewer.layers.save(filename, selected=selected)
                error_messages = "\n".join(
                    [str(x.message.args[0]) for x in wa]
                )

            if not saved:
                raise OSError(
                    trans._(
                        "File {filename} save failed.\n{error_messages}",
                        deferred=True,
                        filename=filename,
                        error_messages=error_messages,
                    )
                )
            else:
                update_save_history(saved[0])

    def _update_welcome_screen(self, event=None):
        """Update welcome screen display based on layer count.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if self._show_welcome_screen:
            self._canvas_overlay.set_welcome_visible(not self.viewer.layers)

    def _screenshot(self, flash=True):
        """Capture a screenshot of the Vispy canvas.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        img = self.canvas.native.grabFramebuffer()
        if flash:
            from .utils import add_flash_animation

            # Here we are actually applying the effect to the `_canvas_overlay`
            # and not # the `native` widget because it does not work on the
            # `native` widget. It's probably because the widget is in a stack
            # with the `QtWelcomeWidget`.
            add_flash_animation(self._canvas_overlay)
        return img

    def screenshot(self, path=None, flash=True):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = QImg2array(self._screenshot(flash))
        if path is not None:
            imsave(path, img)  # scikit-image imsave method
        return img

    def clipboard(self, flash=True):
        """Take a screenshot of the currently displayed screen and copy the
        image to the clipboard.

        Parameters
        ----------
        flash : bool
            Flag to indicate whether flash animation should be shown after
            the screenshot was captured.
        """
        cb = QGuiApplication.clipboard()
        cb.setImage(self._screenshot(flash))

    def _screenshot_dialog(self):
        """Save screenshot of current display, default .png"""
        hist = get_save_history()
        dial = ScreenshotDialog(self.screenshot, self, hist[0], hist)
        if dial.exec_():
            update_save_history(dial.selectedFiles()[0])

    def _open_files_dialog(self):
        """Add files from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        filenames, _ = dlg.getOpenFileNames(
            parent=self,
            caption=trans._('Select file(s)...'),
            directory=hist[0],
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if (filenames != []) and (filenames is not None):
            self.viewer.open(filenames)
            update_open_history(filenames[0])

    def _open_files_dialog_as_stack_dialog(self):
        """Add files as a stack, from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        filenames, _ = dlg.getOpenFileNames(
            parent=self,
            caption=trans._('Select files...'),
            directory=hist[0],  # home dir by default
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if (filenames != []) and (filenames is not None):
            self.viewer.open(filenames, stack=True)
            update_open_history(filenames[0])

    def _open_folder_dialog(self):
        """Add a folder of files from the menubar."""
        dlg = QFileDialog()
        hist = get_open_history()
        dlg.setHistory(hist)

        folder = dlg.getExistingDirectory(
            parent=self,
            caption=trans._('Select folder...'),
            directory=hist[0],  # home dir by default
            options=(
                QFileDialog.DontUseNativeDialog
                if in_ipython()
                else QFileDialog.Options()
            ),
        )

        if folder not in {'', None}:
            self.viewer.open([folder])
            update_open_history(folder)

    def _toggle_chunk_outlines(self):
        """Toggle whether we are drawing outlines around the chunks."""
        from ..layers.image.experimental.octree_image import _OctreeImageBase

        for layer in self.viewer.layers:
            if isinstance(layer, _OctreeImageBase):
                layer.display.show_grid = not layer.display.show_grid

    def _on_interactive(self, event):
        """Link interactive attributes of view and viewer.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self.view.interactive = self.viewer.camera.interactive

    def _on_cursor(self, event):
        """Set the appearance of the mouse cursor.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        cursor = self.viewer.cursor.style
        # Scale size by zoom if needed
        if self.viewer.cursor.scaled:
            size = self.viewer.cursor.size * self.viewer.camera.zoom
        else:
            size = self.viewer.cursor.size

        if cursor == 'square':
            # make sure the square fits within the current canvas
            if size < 8 or size > (
                min(*self.viewer.window.qt_viewer.canvas.size) - 4
            ):
                q_cursor = self._cursors['cross']
            else:
                q_cursor = QCursor(square_pixmap(size))
        elif cursor == 'circle':
            q_cursor = QCursor(circle_pixmap(size))
        else:
            q_cursor = self._cursors[cursor]

        self.canvas.native.setCursor(q_cursor)

    def toggle_console_visibility(self, event=None):
        """Toggle console visible and not visible.

        Imports the console the first time it is requested.
        """
        # force instantiation of console if not already instantiated
        _ = self.console

        viz = not self.dockConsole.isVisible()
        # modulate visibility at the dock widget level as console is dockable
        self.dockConsole.setVisible(viz)
        if self.dockConsole.isFloating():
            self.dockConsole.setFloating(True)

        if viz:
            self.dockConsole.raise_()

        self.viewerButtons.consoleButton.setProperty(
            'expanded', self.dockConsole.isVisible()
        )
        self.viewerButtons.consoleButton.style().unpolish(
            self.viewerButtons.consoleButton
        )
        self.viewerButtons.consoleButton.style().polish(
            self.viewerButtons.consoleButton
        )

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

        position_world = list(self.viewer.dims.point)
        for i, d in enumerate(self.viewer.dims.displayed):
            position_world[d] = position_world_slice[i]

        return tuple(position_world)

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
        bottom_right = self._map_canvas2world(self.canvas.size)
        return np.array([top_left, bottom_right])

    def on_resize(self, event):
        """Called whenever canvas is resized.

        event : vispy.util.event.Event
            The vispy event that triggered this method.
        """
        self.viewer._canvas_size = tuple(self.canvas.size[::-1])

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

    def on_mouse_wheel(self, event):
        """Called whenever mouse wheel activated in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_wheel_callbacks, event)

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

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_press_callbacks, event)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_move_callbacks, event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : vispy.event.Event
            The vispy event that triggered this method.
        """
        self._process_mouse_event(mouse_release_callbacks, event)

    def on_draw(self, event):
        """Called whenever the canvas is drawn.

        This is triggered from vispy whenever new data is sent to the canvas or
        the camera is moved and is connected in the `QtViewer`.
        """
        for layer in self.viewer.layers:
            if layer.ndim <= self.viewer.dims.ndim:
                layer._update_draw(
                    scale_factor=1 / self.viewer.camera.zoom,
                    corner_pixels_displayed=self._canvas_corners_in_world[
                        :, layer._displayed_axes
                    ],
                    shape_threshold=self.canvas.size,
                )

    def set_welcome_visible(self, visible):
        """Show welcome screen widget."""
        self._show_welcome_screen = visible
        self._canvas_overlay.set_welcome_visible(visible)

    def keyPressEvent(self, event):
        """Called whenever a key is pressed.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.canvas._backend._keyEvent(self.canvas.events.key_press, event)
        event.accept()

    def keyReleaseEvent(self, event):
        """Called whenever a key is released.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.canvas._backend._keyEvent(self.canvas.events.key_release, event)
        event.accept()

    def dragEnterEvent(self, event):
        """Ignore event if not dragging & dropping a file or URL to open.

        Using event.ignore() here allows the event to pass through the
        parent widget to its child widget, otherwise the parent widget
        would catch the event and not pass it on to the child widget.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Add local files and web URLS with drag and drop.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        shift_down = QGuiApplication.keyboardModifiers() & Qt.ShiftModifier
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                filenames.append(url.toLocalFile())
            else:
                filenames.append(url.toString())

        self.viewer.open(filenames, stack=bool(shift_down))

    def closeEvent(self, event):
        """Cleanup and close.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self.layers.close()

        # if the viewer.QtDims object is playing an axis, we need to terminate
        # the AnimationThread before close, otherwise it will cauyse a segFault
        # or Abort trap. (calling stop() when no animation is occurring is also
        # not a problem)
        self.dims.stop()
        self.canvas.native.deleteLater()
        if self._console is not None:
            self.console.close()
        self.dockConsole.deleteLater()
        event.accept()


if TYPE_CHECKING:
    from ..components.experimental.remote import RemoteManager
    from .experimental.qt_poll import QtPoll


def _create_qt_poll(parent: QObject, camera: Camera) -> Optional[QtPoll]:
    """Create and return a QtPoll instance, if needed.

    Create a QtPoll instance for octree or monitor.

    Octree needs QtPoll so VispyTiledImageLayer can finish in-progress
    loads even if the camera is not moving. Once loading is finish it will
    tell QtPoll it no longer needs to be polled.

    Monitor needs QtPoll to poll for incoming messages. This might be
    temporary until we can process incoming messages with a dedicated
    thread.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    camera : Camera
        Camera that the QtPoll object will listen to.

    Returns
    -------
    Optional[QtPoll]
        The new QtPoll instance, if we need one.
    """
    if not config.async_octree and not config.monitor:
        return None

    from .experimental.qt_poll import QtPoll

    qt_poll = QtPoll(parent)
    camera.events.connect(qt_poll.on_camera)
    return qt_poll


def _create_remote_manager(
    layers: LayerList, qt_poll
) -> Optional[RemoteManager]:
    """Create and return a RemoteManager instance, if we need one.

    Parameters
    ----------
    layers : LayersList
        The viewer's layers.
    qt_poll : QtPoll
        The viewer's QtPoll instance.
    """
    if not config.monitor:
        return None  # Not using the monitor at all

    from ..components.experimental.monitor import monitor
    from ..components.experimental.remote import RemoteManager

    # Start the monitor so we can access its events. The monitor has no
    # dependencies to napari except to utils.Event.
    started = monitor.start()

    if not started:
        return None  # Probably not >= Python 3.9, so no manager is needed.

    # Create the remote manager and have monitor call its process_command()
    # method to execute commands from clients.
    manager = RemoteManager(layers)

    # RemoteManager will process incoming command from the monitor.
    monitor.run_command_event.connect(manager.process_command)

    # QtPoll should pool the RemoteManager and the Monitor.
    qt_poll.events.poll.connect(manager.on_poll)
    qt_poll.events.poll.connect(monitor.on_poll)

    return manager
