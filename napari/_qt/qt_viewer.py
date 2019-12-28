import os.path
import inspect
from pathlib import Path

from qtpy import QtGui
from qtpy.QtCore import QCoreApplication, Qt, QSize
from qtpy.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QSplitter
from qtpy.QtGui import QCursor, QPixmap
from qtpy.QtCore import QThreadPool
from qtpy import API_NAME
from vispy.scene import SceneCanvas, PanZoomCamera, ArcballCamera
from vispy.visuals.transforms import ChainTransform
from vispy.app import use_app

from .qt_dims import QtDims
from .qt_layerlist import QtLayerList
from ..resources import stylesheet
from ..utils.theme import template
from ..utils.misc import str_to_rgb
from ..utils.interactions import (
    ReadOnlyWrapper,
    mouse_press_callbacks,
    mouse_move_callbacks,
    mouse_release_callbacks,
)
from ..utils.keybindings import components_to_key_combo

from .utils import QImg2array
from .qt_controls import QtControls
from .qt_viewer_buttons import QtLayerButtons, QtViewerButtons
from .qt_console import QtConsole
from .qt_viewer_dock_widget import QtViewerDockWidget
from .qt_about_keybindings import QtAboutKeybindings
from .._vispy import create_vispy_visual


# set vispy application to the appropriate qt backend
use_app(API_NAME)


class QtViewer(QSplitter):
    def __init__(self, viewer):
        super().__init__()

        self.pool = QThreadPool()

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer
        self.dims = QtDims(self.viewer.dims)
        self.controls = QtControls(self.viewer)
        self.layers = QtLayerList(self.viewer.layers)
        self.layerButtons = QtLayerButtons(self.viewer)
        self.viewerButtons = QtViewerButtons(self.viewer)
        self.console = QtConsole({'viewer': self.viewer})

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
            self.console,
            name='console',
            area='bottom',
            allowed_areas=['top', 'bottom'],
            shortcut='Ctrl+Shift+C',
        )
        self.dockConsole.setVisible(False)
        self.dockLayerControls.visibilityChanged.connect(self._constrain_width)
        self.dockLayerList.setMaximumWidth(258)
        self.dockLayerList.setMinimumWidth(258)

        self.aboutKeybindings = QtAboutKeybindings(self.viewer)
        self.aboutKeybindings.hide()

        # This dictionary holds the corresponding vispy visual for each layer
        self.layer_to_visual = {}

        if self.console.shell is not None:
            self.viewerButtons.consoleButton.clicked.connect(
                lambda: self.toggle_console()
            )
        else:
            self.viewerButtons.consoleButton.setEnabled(False)

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.canvas.events.ignore_callback_errors = False
        self.canvas.events.draw.connect(self.dims.enable_play)
        self.canvas.native.setMinimumSize(QSize(200, 200))
        self.canvas.context.set_depth_func('lequal')

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_key_release)
        self.canvas.connect(self.on_draw)

        self.view = self.canvas.central_widget.add_view()
        self._update_camera()

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
            'disabled': QCursor(
                QPixmap(':/icons/cursor/cursor_disabled.png').scaled(20, 20)
            ),
            'cross': Qt.CrossCursor,
            'forbidden': Qt.ForbiddenCursor,
            'pointing': Qt.PointingHandCursor,
            'standard': QCursor(),
        }

        self._update_palette(viewer.palette)

        self._key_release_generators = {}

        self.viewer.events.interactive.connect(self._on_interactive)
        self.viewer.events.cursor.connect(self._on_cursor)
        self.viewer.events.reset_view.connect(self._on_reset_view)
        self.viewer.events.palette.connect(
            lambda event: self._update_palette(event.palette)
        )
        self.viewer.layers.events.reordered.connect(self._reorder_layers)
        self.viewer.layers.events.added.connect(self._add_layer)
        self.viewer.layers.events.removed.connect(self._remove_layer)
        self.viewer.dims.events.camera.connect(
            lambda event: self._update_camera()
        )
        # stop any animations whenever the layers change
        self.viewer.events.layers_change.connect(lambda x: self.dims.stop())

        self.setAcceptDrops(True)

    def _constrain_width(self, event):
        # allow the layer controls to be wider, only if floated
        if self.dockLayerControls.isFloating():
            self.controls.setMaximumWidth(700)
        else:
            self.controls.setMaximumWidth(220)

    def _add_layer(self, event):
        """When a layer is added, set its parent and order."""
        layers = event.source
        layer = event.item
        vispy_layer = create_vispy_visual(layer)
        vispy_layer.camera = self.view.camera
        vispy_layer.node.parent = self.view.scene
        vispy_layer.order = len(layers)
        self.layer_to_visual[layer] = vispy_layer

    def _remove_layer(self, event):
        """When a layer is removed, remove its parent."""
        layer = event.item
        vispy_layer = self.layer_to_visual[layer]
        vispy_layer.node.transforms = ChainTransform()
        vispy_layer.node.parent = None
        del self.layer_to_visual[layer]

    def _reorder_layers(self, event):
        """When the list is reordered, propagate changes to draw order."""
        for i, layer in enumerate(self.viewer.layers):
            vispy_layer = self.layer_to_visual[layer]
            vispy_layer.order = i
        self.canvas._draw_order.clear()
        self.canvas.update()

    def _update_camera(self):
        if self.viewer.dims.ndisplay == 3:
            # Set a 3D camera
            if not isinstance(self.view.camera, ArcballCamera):
                self.view.camera = ArcballCamera(name="ArcballCamera", fov=0)
                # flip y-axis to have correct alignment
                # self.view.camera.flip = (0, 1, 0)

                self.view.camera.viewbox_key_event = viewbox_key_event
                self.viewer.reset_view()
        else:
            # Set 2D camera
            if not isinstance(self.view.camera, PanZoomCamera):
                self.view.camera = PanZoomCamera(
                    aspect=1, name="PanZoomCamera"
                )
                # flip y-axis to have correct alignment
                self.view.camera.flip = (0, 1, 0)

                self.view.camera.viewbox_key_event = viewbox_key_event
                self.viewer.reset_view()

    def screenshot(self):
        """Take currently displayed screen and convert to an image array.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        img = self.canvas.native.grabFramebuffer()
        return QImg2array(img)

    def _open_images(self):
        """Add image files from the menubar."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select image(s)...',
            directory=self._last_visited_dir,  # home dir by default
        )
        if (filenames != []) and (filenames is not None):
            self._add_files(filenames)

    def _open_folder(self):
        """Add a folder of files from the menubar."""
        folder = QFileDialog.getExistingDirectory(
            parent=self,
            caption='Select folder...',
            directory=self._last_visited_dir,  # home dir by default
        )
        if folder not in {'', None}:
            self._add_files([folder])

    def _add_files(self, filenames):
        """Add an image layer to the viewer.

        If multiple images are selected, they are stacked along the 0th
        axis.

        Parameters
        -------
        filenames : list
            List of filenames to be opened
        """
        if len(filenames) > 0:
            self.viewer.add_image(path=filenames)
            self._last_visited_dir = os.path.dirname(filenames[0])

    def _on_interactive(self, event):
        self.view.interactive = self.viewer.interactive

    def _on_cursor(self, event):
        cursor = self.viewer.cursor
        size = self.viewer.cursor_size
        if cursor == 'square':
            if size < 10 or size > 300:
                q_cursor = self._cursors['cross']
            else:
                q_cursor = QCursor(
                    QPixmap(':/icons/cursor/cursor_square.png').scaledToHeight(
                        size
                    )
                )
        else:
            q_cursor = self._cursors[cursor]
        self.canvas.native.setCursor(q_cursor)

    def _on_reset_view(self, event):
        if isinstance(self.view.camera, ArcballCamera):
            quat = self.view.camera._quaternion.create_from_axis_angle(
                *event.quaternion
            )
            self.view.camera._quaternion = quat
            self.view.camera.center = event.center
            self.view.camera.scale_factor = event.scale_factor
        else:
            # Assumes default camera has the same properties as PanZoomCamera
            self.view.camera.rect = event.rect

    def _update_palette(self, palette):
        # template and apply the primary stylesheet
        themed_stylesheet = template(stylesheet, **palette)
        self.console.style_sheet = themed_stylesheet
        self.console.syntax_style = palette['syntax_style']
        bracket_color = QtGui.QColor(*str_to_rgb(palette['highlight']))
        self.console._bracket_matcher.format.setBackground(bracket_color)
        self.setStyleSheet(themed_stylesheet)
        self.aboutKeybindings.setStyleSheet(themed_stylesheet)
        self.canvas.bgcolor = palette['canvas']

    def toggle_console(self):
        """Toggle console visible and not visible."""
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

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if event.pos is None:
            return

        event = ReadOnlyWrapper(event)
        mouse_press_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            # Line bellow needed until layer mouse callbacks are refactored
            self.layer_to_visual[layer].on_mouse_press(event)
            mouse_press_callbacks(layer, event)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return

        mouse_move_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            # Line bellow needed until layer mouse callbacks are refactored
            self.layer_to_visual[layer].on_mouse_move(event)
            mouse_move_callbacks(layer, event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        mouse_release_callbacks(self.viewer, event)

        layer = self.viewer.active_layer
        if layer is not None:
            # Line bellow needed until layer mouse callbacks are refactored
            self.layer_to_visual[layer].on_mouse_release(event)
            mouse_release_callbacks(layer, event)

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        if (
            event.native is not None
            and event.native.isAutoRepeat()
            and event.key.name not in ['Up', 'Down', 'Left', 'Right']
        ) or event.key is None:
            # pass is no key is present or if key is held down, unless the
            # key being held down is one of the navigation keys
            return

        comb = components_to_key_combo(event.key.name, event.modifiers)

        layer = self.viewer.active_layer

        if layer is not None and comb in layer.keymap:
            parent = layer
        elif comb in self.viewer.keymap:
            parent = self.viewer
        else:
            return

        func = parent.keymap[comb]
        gen = func(parent)

        if inspect.isgeneratorfunction(func):
            try:
                next(gen)
            except StopIteration:  # only one statement
                pass
            else:
                self._key_release_generators[event.key] = gen

    def on_key_release(self, event):
        """Called whenever key released in canvas.
        """
        try:
            next(self._key_release_generators[event.key])
        except (KeyError, StopIteration):
            pass

    def on_draw(self, event):
        """Called whenever drawn in canvas. Called for all layers, not just top
        """
        for visual in self.layer_to_visual.values():
            visual.on_draw(event)

    def keyPressEvent(self, event):
        self.canvas._backend._keyEvent(self.canvas.events.key_press, event)
        event.accept()

    def keyReleaseEvent(self, event):
        self.canvas._backend._keyEvent(self.canvas.events.key_release, event)
        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Add local files and web URLS with drag and drop."""
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                filenames.append(url.toLocalFile())
            else:
                filenames.append(url.toString())
        self._add_files(filenames)

    def closeEvent(self, event):
        if self.pool.activeThreadCount() > 0:
            self.pool.clear()
        event.accept()

    def shutdown(self):
        self.pool.clear()
        self.canvas.close()
        self.console.shutdown()


def viewbox_key_event(event):
    """ViewBox key event handler
    Parameters
    ----------
    event : instance of Event
        The event.
    """
    return
