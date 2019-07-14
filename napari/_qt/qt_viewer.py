import os.path
from glob import glob
import inspect
from pathlib import Path

from qtpy.QtCore import QCoreApplication, Qt, QSize
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter, QFileDialog
from qtpy.QtGui import QCursor, QPixmap
from qtpy import API_NAME
from vispy.scene import SceneCanvas, PanZoomCamera
from vispy.app import use_app

from .qt_dims import QtDims
from .qt_layerlist import QtLayerList
from ..resources import resources_dir
from ..util.theme import template
from ..util.misc import is_multichannel
from ..util.keybindings import components_to_key_combo
from ..util.io import read

from .qt_controls import QtControls
from .qt_layer_buttons import QtLayersButtons


# set vispy application to the appropriate qt backend
use_app(API_NAME)


class QtViewer(QSplitter):
    with open(os.path.join(resources_dir, 'stylesheet.qss'), 'r') as f:
        raw_stylesheet = f.read()

    def __init__(self, viewer):
        super().__init__()

        QCoreApplication.setAttribute(
            Qt.AA_UseStyleSheetPropagationInWidgetStyles, True
        )

        self.viewer = viewer

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(100, 100))

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_key_release)
        self.canvas.connect(self.on_draw)

        self.view = self.canvas.central_widget.add_view()

        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()
        self.view.camera.viewbox_key_event = viewbox_key_event

        # TO DO: Remove
        self.viewer._view = self.view

        center = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(15, 20, 15, 10)
        center_layout.addWidget(self.canvas.native)
        self.dims = QtDims(self.viewer.dims)
        center_layout.addWidget(self.dims)
        center.setLayout(center_layout)

        # Add controls, center, and layerlist
        self.control_panel = QtControls(viewer)
        self.addWidget(self.control_panel)
        self.addWidget(center)

        right = QWidget()
        right_layout = QVBoxLayout()
        self.layers = QtLayerList(self.viewer.layers)
        right_layout.addWidget(self.layers)
        self.buttons = QtLayersButtons(viewer)
        right_layout.addWidget(self.buttons)
        right.setLayout(right_layout)
        right.setMinimumSize(QSize(308, 250))

        self.addWidget(right)

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
        self.viewer.layers.events.reordered.connect(self._update_canvas)

        self.setAcceptDrops(True)

    def screenshot(self, region=None, size=None, bgcolor=None):
        """Render the scene to an offscreen buffer and return the image array.

        Parameters
        ----------
        region : tuple | None
            Specifies the region of the canvas to render. Format is
            (x, y, w, h). By default, the entire canvas is rendered.
        size : tuple | None
            Specifies the size of the image array to return. If no size is
            given, then the size of the *region* is used, multiplied by the
            pixel scaling factor of the canvas (see `pixel_scale`). This
            argument allows the scene to be rendered at resolutions different
            from the native canvas resolution.
        bgcolor : instance of Color | None
            The background color to use.

        Returns
        -------
        image : array
            Numpy array of type ubyte and shape (h, w, 4). Index [0, 0] is the
            upper-left corner of the rendered region.
        """
        return self.canvas.render(region, size, bgcolor)

    def _open_images(self):
        """Adds image files from the menubar."""
        filenames, _ = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select image(s)...',
            directory=self._last_visited_dir,  # home dir by default
        )
        self._add_files(filenames)

    def _add_files(self, filenames):
        """Adds an image layer to the viewer.

        Whether the image is multichannel is determined by
        :func:`napari.util.misc.is_multichannel`.

        If multiple images are selected, they are stacked along the 0th
        axis.

        Parameters
        -------
        filenames : list
            List of filenames to be opened
        """
        if len(filenames) > 0:
            image = read(filenames)
            self.viewer.add_image(
                image, multichannel=is_multichannel(image.shape)
            )
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
        self.view.camera.rect = event.viewbox

    def _update_canvas(self, event):
        """Clears draw order and refreshes canvas. Usefeul for when layers are
        reoredered.
        """
        self.canvas._draw_order.clear()
        self.canvas.update()

    def _update_palette(self, palette):
        # template and apply the primary stylesheet
        themed_stylesheet = template(self.raw_stylesheet, **palette)
        self.setStyleSheet(themed_stylesheet)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_move(event)

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_press(event)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_mouse_release(event)

    def on_key_press(self, event):
        """Called whenever key pressed in canvas.
        """
        if event.native.isAutoRepeat() or event.key is None:
            return

        comb = components_to_key_combo(event.key.name, event.modifiers)

        layer = self.viewer.active_layer

        # TODO: remove me once keybinding system converted
        if layer is not None:
            layer.on_key_press(event)

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

        # TODO: remove me once keybinding system converted
        layer = self.viewer.active_layer
        if layer is not None:
            layer.on_key_release(event)

    def on_draw(self, event):
        """Called whenever drawn in canvas. Called for all layers, not just top
        """
        for layer in self.viewer.layers:
            layer.on_draw(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Add local files and web URLS with drag and drop."""
        filenames = []
        for url in event.mimeData().urls():
            path = url.toString()
            if os.path.isfile(path):
                filenames.append(path)
            elif os.path.isdir(path):
                filenames = filenames + list(glob(os.path.join(path, '*')))
            else:
                filenames.append(path)
        self._add_files(filenames)


def viewbox_key_event(event):
    """ViewBox key event handler
    Parameters
    ----------
    event : instance of Event
        The event.
    """
    return
