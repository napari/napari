import os.path
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import QCoreApplication, QObject, QSize, Qt
from qtpy.QtGui import QCursor, QGuiApplication
from qtpy.QtWidgets import QFileDialog, QSplitter, QVBoxLayout, QWidget

from ..components.camera import Camera
from ..resources import get_stylesheet
from ..utils import config, perf
from ..utils.interactions import (
    ReadOnlyWrapper,
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
    mouse_wheel_callbacks,
)
from ..utils.io import imsave
from ..utils.key_bindings import components_to_key_combo
from ..utils.theme import template
from .dialogs.qt_about_key_bindings import QtAboutKeyBindings
from .dialogs.screenshot_dialog import ScreenshotDialog
from .tracing.qt_performance import QtPerformance
from .utils import QImg2array, circle_pixmap, square_pixmap
from .widgets.qt_dims import QtDims
from .widgets.qt_layerlist import QtLayerList
from .widgets.qt_viewer_buttons import QtLayerButtons, QtViewerButtons
from .widgets.qt_viewer_dock_widget import QtViewerDockWidget

from .._vispy import (  # isort:skip
    VispyAxesVisual,
    VispyCamera,
    VispyCanvas,
    VispyScaleBarVisual,
    VispyWelcomeVisual,
    create_vispy_visual,
)


class QtViewer(QSplitter):
    """Qt view for the napari Viewer model.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Napari viewer containing the rendered scene, layers, and controls.
    welcome : bool
        Flag to show a welcome message when no layers are present in the
        canvas.

    Attributes
    ----------
    canvas : vispy.scene.SceneCanvas
        Canvas for rendering the current view.
    console : QtConsole
        iPython console terminal integrated into the napari GUI.
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

    raw_stylesheet = get_stylesheet()

    def __init__(self, viewer, welcome=False):

        # Avoid circular import.
        from .layer_controls import QtLayerControlsContainer

        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer
        self.dims = QtDims(self.viewer.dims)
        self.controls = QtLayerControlsContainer(self.viewer)
        self.layers = QtLayerList(self.viewer.layers)
        self.layerButtons = QtLayerButtons(self.viewer)
        self.viewerButtons = QtViewerButtons(self.viewer)
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
            name='layer list',
            area='left',
            allowed_areas=['left', 'right'],
        )
        self.dockLayerControls = QtViewerDockWidget(
            self,
            self.controls,
            name='layer controls',
            area='left',
            allowed_areas=['left', 'right'],
        )
        self.dockConsole = QtViewerDockWidget(
            self,
            QWidget(),
            name='console',
            area='bottom',
            allowed_areas=['top', 'bottom'],
            shortcut='Ctrl+Shift+C',
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
        self.viewerButtons.consoleButton.clicked.connect(
            self.toggle_console_visibility
        )

        self._create_canvas()

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 22, 10, 2)
        main_layout.addWidget(self.canvas.native)
        main_layout.addWidget(self.dims)
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)

        self.setOrientation(Qt.Vertical)
        self.addWidget(main_widget)

        self._last_visited_dir = str(Path.home())

        self._cursors = {
            'cross': Qt.CrossCursor,
            'forbidden': Qt.ForbiddenCursor,
            'pointing': Qt.PointingHandCursor,
            'standard': QCursor(),
        }

        self._update_palette()

        self.viewer.events.interactive.connect(self._on_interactive)
        self.viewer.cursor.events.style.connect(self._on_cursor)
        self.viewer.cursor.events.size.connect(self._on_cursor)
        self.viewer.events.palette.connect(self._update_palette)
        self.viewer.layers.events.reordered.connect(self._reorder_layers)
        self.viewer.layers.events.inserted.connect(self._on_add_layer_change)
        self.viewer.layers.events.removed.connect(self._remove_layer)

        # stop any animations whenever the layers change
        self.viewer.events.layers_change.connect(lambda x: self.dims.stop())

        self.setAcceptDrops(True)

        for layer in self.viewer.layers:
            self._add_layer(layer)

        self.view = self.canvas.central_widget.add_view()
        self.camera = VispyCamera(
            self.view, self.viewer.camera, self.viewer.dims
        )
        self.canvas.connect(self.camera.on_draw)

        # Add axes, scale bar and welcome visuals.
        self._add_visuals(welcome)

        # Optional experimental monitor service.
        self._qt_monitor = _create_qt_monitor(self, self.viewer.camera)

        # Experimental QtPool for Octree visuals.
        self._qt_poll = _create_qt_poll(self, self.viewer.camera)

    def _create_canvas(self) -> None:
        """Create the canvas and hook up events."""
        self.canvas = VispyCanvas(
            keys=None,
            vsync=True,
            parent=self,
            size=self.viewer._canvas_size[::-1],
        )
        self.canvas.events.ignore_callback_errors = False
        self.canvas.events.draw.connect(self.dims.enable_play)
        self.canvas.native.setMinimumSize(QSize(200, 200))
        self.canvas.context.set_depth_func('lequal')

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_key_release)
        self.canvas.connect(self.on_mouse_wheel)
        self.canvas.connect(self.on_draw)
        self.canvas.connect(self.on_resize)

    def _add_visuals(self, welcome: bool) -> None:
        """Add visuals for axes, scale bar, and welcome text.

        Parameters
        ----------
        welcome : bool
            Show the welcome visual.
        """

        self.axes = VispyAxesVisual(
            self.viewer.axes,
            self.viewer.camera,
            self.viewer.dims,
            parent=self.view.scene,
            order=1e6,
        )
        self.scale_bar = VispyScaleBarVisual(
            self.viewer.scale_bar,
            self.viewer.camera,
            parent=self.view,
            order=1e6 + 1,
        )
        self.canvas.events.resize.connect(self.scale_bar._on_position_change)

        self._show_welcome = welcome and config.allow_welcome_visual
        if self._show_welcome:
            self.welcome = VispyWelcomeVisual(
                self.viewer, parent=self.view, order=-100
            )
            self.viewer.events.layers_change.connect(
                self.welcome._on_visible_change
            )
            self.viewer.events.palette.connect(self.welcome._on_palette_change)
            self.canvas.events.resize.connect(self.welcome._on_canvas_change)

    def _create_performance_dock_widget(self):
        """Create the dock widget that shows performance metrics.
        """
        if perf.USE_PERFMON:
            return QtViewerDockWidget(
                self,
                QtPerformance(),
                name='performance',
                area='bottom',
                shortcut='Ctrl+Shift+P',
            )
        return None

    @property
    def console(self):
        """QtConsole: iPython console terminal integrated into the napari GUI.
        """
        if self._console is None:
            from .widgets.qt_console import QtConsole

            self.console = QtConsole({'viewer': self.viewer})
        return self._console

    @console.setter
    def console(self, console):
        self._console = console
        self.dockConsole.widget = console
        self._update_palette()

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

        if self._qt_poll is not None:
            # Visuals might need to be polled when the camera moves and during
            # a short period after the movement stops.
            self._qt_poll.events.poll.connect(vispy_layer._on_poll)

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
            msg = "There are no layers in the viewer to save"
        elif selected and not len(self.viewer.layers.selected):
            msg = (
                'Please select one or more layers to save,'
                '\nor use "Save all layers..."'
            )
        if msg:
            raise IOError("Nothing to save")

        filename, _ = QFileDialog.getSaveFileName(
            parent=self,
            caption=f'Save {"selected" if selected else "all"} layers',
            directory=self._last_visited_dir,  # home dir by default
        )

        if filename:
            with warnings.catch_warnings(record=True) as wa:
                saved = self.viewer.layers.save(filename, selected=selected)
                error_messages = "\n".join(
                    [str(x.message.args[0]) for x in wa]
                )
            if not saved:
                raise IOError(
                    f"File {filename} save failed.\n{error_messages}"
                )

    def screenshot(self, path=None):
        """Take currently displayed screen and convert to an image array.

        Parameters
        ----------
        path : str
            Filename for saving screenshot image.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = QImg2array(self.canvas.native.grabFramebuffer())
        if path is not None:
            imsave(path, img)  # scikit-image imsave method
        return img

    def _screenshot_dialog(self):
        """Save screenshot of current display, default .png"""
        dial = ScreenshotDialog(self.screenshot, self, self._last_visited_dir)
        if dial.exec_():
            self._last_visited_dir = os.path.dirname(dial.selectedFiles()[0])

    def _open_files_dialog(self):
        """Add files from the menubar."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select file(s)...',
            directory=self._last_visited_dir,  # home dir by default
        )
        if (filenames != []) and (filenames is not None):
            self.viewer.open(filenames)

    def _open_files_dialog_as_stack_dialog(self):
        """Add files as a stack, from the menubar."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select files...',
            directory=self._last_visited_dir,  # home dir by default
        )
        if (filenames != []) and (filenames is not None):
            self.viewer.open(filenames, stack=True)

    def _open_folder_dialog(self):
        """Add a folder of files from the menubar."""
        folder = QFileDialog.getExistingDirectory(
            parent=self,
            caption='Select folder...',
            directory=self._last_visited_dir,  # home dir by default
        )
        if folder not in {'', None}:
            self.viewer.open([folder])

    def _toggle_chunk_outlines(self):
        """Toggle whether we are drawing outlines around the chunks."""
        from ..layers.image.experimental.octree_image import OctreeImage

        for layer in self.viewer.layers:
            if isinstance(layer, OctreeImage):
                layer.display.show_grid = not layer.display.show_grid

    def _on_interactive(self, event):
        """Link interactive attributes of view and viewer.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        self.view.interactive = self.viewer.interactive

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

    def _update_palette(self, event=None):
        """Update the napari GUI theme."""
        # template and apply the primary stylesheet
        themed_stylesheet = template(
            self.raw_stylesheet, **self.viewer.palette
        )
        if self._console is not None:
            self.console._update_palette(
                self.viewer.palette, themed_stylesheet
            )
        self.setStyleSheet(themed_stylesheet)
        self.canvas.bgcolor = self.viewer.palette['canvas']

    def toggle_console_visibility(self, event=None):
        """Toggle console visible and not visible.

        Imports the console the first time it is requested.
        """
        # force instantiation of console if not already instantiated
        _ = self.console

        viz = not self.dockConsole.isVisible()
        # modulate visibility at the dock widget level as console is docakable
        self.dockConsole.setVisible(viz)
        if self.dockConsole.isFloating():
            self.dockConsole.setFloating(True)

        self.viewerButtons.consoleButton.setProperty(
            'expanded', self.dockConsole.isVisible()
        )
        self.viewerButtons.consoleButton.style().unpolish(
            self.viewerButtons.consoleButton
        )
        self.viewerButtons.consoleButton.style().polish(
            self.viewerButtons.consoleButton
        )

    def show_key_bindings_dialog(self, event=None):
        dialog = QtAboutKeyBindings(self.viewer, parent=self)
        dialog.show()

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
        transform = self.view.camera.transform.inverse
        mapped_position = transform.map(list(position))[:nd]
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

    def on_mouse_wheel(self, event):
        """Called whenever mouse wheel activated in canvas.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
        """
        if event.pos is None:
            return

        event = ReadOnlyWrapper(event)
        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))
        mouse_wheel_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            mouse_wheel_callbacks(layer, event)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.pos is None:
            return

        event = ReadOnlyWrapper(event)
        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))
        mouse_press_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            mouse_press_callbacks(layer, event)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.pos is None:
            return

        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))
        mouse_move_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            mouse_move_callbacks(layer, event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.pos is None:
            return

        self.viewer.cursor.position = self._map_canvas2world(list(event.pos))
        mouse_release_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            mouse_release_callbacks(layer, event)

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if (
            event.native is not None
            and event.native.isAutoRepeat()
            and event.key.name not in ['Up', 'Down', 'Left', 'Right']
        ) or event.key is None:
            # pass if no key is present or if key is held down, unless the
            # key being held down is one of the navigation keys
            # this helps for scrolling, etc.
            return

        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.viewer.press_key(combo)

    def on_key_release(self, event):
        """Called whenever key released in canvas.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        if event.key is None or (
            # on linux press down is treated as multiple press and release
            event.native is not None
            and event.native.isAutoRepeat()
        ):
            return
        combo = components_to_key_combo(event.key.name, event.modifiers)
        self.viewer.release_key(combo)

    def on_draw(self, event):
        """Called whenever the canvas is drawn.

        This is triggered from vispy whenever new data is sent to the canvas or
        the camera is moved and is connected in the `QtViewer`.
        """
        for layer in self.viewer.layers:
            if layer.ndim <= self.viewer.dims.ndim:
                layer._update_draw(
                    scale_factor=1 / self.viewer.camera.zoom,
                    corner_pixels=self._canvas_corners_in_world[
                        :, -layer.ndim :
                    ],
                    shape_threshold=self.canvas.size,
                )

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
    from .experimental.qt_monitor import QtMonitor
    from .experimental.qt_poll import QtPoll


def _create_qt_monitor(
    parent: QObject, camera: Camera
) -> 'Optional[QtMonitor]':
    """Create and return a QtMonitor instance.

    A monitor instance is only created if NAPARI_MON is set.

    Parameters
    ----------
    parent : QObject
        Parent for the qtMonitor.
    camera : VispyCamera
        Camera that the monitor will tie into.

    Return
    ------
    Optional[QtMonitor]
        The new monitor instance, if any
    """
    if config.monitor:
        from .experimental.qt_monitor import QtMonitor

        # We can probably get rid of QtMonitor and use QtPoll instead,
        # since it's the same idea but more general.
        return QtMonitor(parent, camera)

    return None


def _create_qt_poll(parent: QObject, camera: Camera) -> 'Optional[QtPoll]':
    """Create and return a QtPoll instance, if enabled.

    A QtPoll instance is only created if using octree for now.

    Parameters
    ----------
    parent : QObject
        Parent for the qtMonitor.
    camera : VispyCamera
        Camera that the monitor will tie into.

    Return
    ------
    Optional[QtMonitor]
        The new monitor instance, if any
    """
    if config.async_octree:
        from .experimental.qt_poll import QtPoll

        return QtPoll(parent, camera)

    return None
