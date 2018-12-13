from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QCursor, QPixmap
from .qt import QtViewer

from numpy import clip, integer, ndarray, append, insert, delete
from copy import copy


from os.path import dirname, join, realpath
dir_path = dirname(realpath(__file__))
path_cursor = join(dir_path,'qt','icons','cursor_disabled.png')

class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders.

    Parameters
    ----------
    parent : Window
        Parent window.

    Attributes
    ----------
    gui : Gui
        Parent gui window.
    layers : LayerList
        List of contained layers.
    dimensions : Dimensions
        Contains view, canvas, axes, indices, dimensions and sliders.
    controlBars : ControlBars
        Contains contorl bar sliders.
    camera : vispy.scene.Camera
        Viewer camera.
    """
    def __init__(self, gui):
        from ._layer_list import LayerList
        from ._control_bars import ControlBars
        from ._dimensions import Dimensions

        self.gui = gui
        self.dimensions = Dimensions(self)
        self.layers = LayerList(self)
        self.controlBars = ControlBars(self)

        self._qt = QtViewer(self)
        self._update = self.dimensions._update
        self._statusBar = self.gui._qt_window.statusBar

        self._qt.canvas.connect(self.on_mouse_move)
        self._qt.canvas.connect(self.on_mouse_press)
        self._qt.canvas.connect(self.on_key_press)
        self._qt.canvas.connect(self.on_key_release)

        self.annotation = False
        self._annotation_history = False
        self._active_image = None
        self._active_markers = None
        self._visible_markers = []

        self._status_widget = QLabel('hold <space> to pan/zoom')
        self._statusBar().addPermanentWidget(self._status_widget)
        self._status_widget.hide()

        self._disabled_cursor = QCursor(QPixmap(path_cursor).scaled(20,20))

    @property
    def _canvas(self):
        return self._qt.canvas

    @property
    def _view(self):
        return self._qt.view

    @property
    def camera(self):
        """vispy.scene.Camera: Viewer camera.
        """
        return self._view.camera
    def reset_view(self):
        """Resets the camera's view.
        """
        try:
            self.camera.set_range()
        except AttributeError:
            pass

    def screenshot(self, *args, **kwargs):
        """Renders the current canvas.

        Returns
        -------
        screenshot : np.ndarray
            View of the current canvas.
        """
        return self._canvas.render(*args, **kwargs)

    def add_layer(self, layer):
        """Adds a layer to the viewer.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.reset_view()

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
        layer : Image
            Layer for the image.
        """
        meta = guess_metadata(image, meta, multichannel, kwargs)

        return self.add_image(image, meta)

    def _update_layers(self):
        """Updates the contained layers.
        """
        for layer in self.layers:
            layer._set_view_slice(self.dimensions.indices)

        self._update_statusBar()

    def _on_layers_change(self, event):
        """Called whenever a layer is changed.
        """
        self.dimensions._child_layer_changed = True
        self.dimensions._update()

    def update_statusBar(self):
        from ..layers._image_layer import Image
        from ..layers._markers_layer import Markers

        msg = '(%d, %d' % (self.dimensions._pos[0], self.dimensions._pos[1])
        if self.dimensions.max_dims > 2:
            for i in range(2,self.dimensions.max_dims):
                msg = msg + ', %d' % self.dimensions.indices[i]
        msg = msg + ')'

    def _set_annotation_mode(self, bool):
        if bool:
            self.annotation = True
            self._qt.view.interactive = False
            if self._active_markers:
                self._qt.canvas.native.setCursor(Qt.CrossCursor)
            else:
                self._qt.canvas.native.setCursor(self._disabled_cursor)
            self._status_widget.show()
        else:
            self.annotation = False
            self._qt.view.interactive = True
            self._qt.canvas.native.setCursor(QCursor())
            self._status_widget.hide()
        self._update_statusBar()

    def _update_active_layers(self):
        from ..layers._image_layer import Image
        from ..layers._markers_layer import Markers
        top_markers = []
        for i, layer in enumerate(self.layers[::-1]):
            if layer.visible and isinstance(layer, Image):
                top_image = len(self.layers) - 1 - i
                break
            elif layer.visible and isinstance(layer, Markers):
                top_markers.append(len(self.layers) - 1 - i)
                coord = [self.dimensions._index[1],self.dimensions._index[0],*self.dimensions._index[2:]]
                layer._set_selected_markers(coord)
        else:
            top_image = None

        active_markers = None
        for i in top_markers:
            if self.layers[i].selected:
                active_markers = i
                break

        self._active_image = top_image
        self._visible_markers = top_markers
        self._active_markers = active_markers

    def _update_statusBar(self):
        msg = '('
        for i in range(0,self.dimensions.max_dims):
            msg = msg + '%d, ' % self.dimensions._index[i]
        msg = msg[:-2]
        msg = msg + ')'

        index = None
        for i in self._visible_markers:
            index = self.layers[i]._selected_markers
            if index is None:
                pass
            else:
                msg = msg + ', %s, index %d' % (self.layers[i].name, index)
                break

        if self._active_image is None:
            pass
        elif index is None:
            msg = msg + ', %s' % self.layers[self._active_image].name
            value = self.layers[self._active_image]._slice_image(self.dimensions._index)
            msg = msg + ', value '
            if isinstance(value, ndarray):
                if isinstance(value[0], integer):
                    msg = msg + '(%d, %d, %d)' % (value[0], value[1], value[2])
                else:
                    msg = msg + '(%.3f, %.3f, %.3f)' % (value[0], value[1], value[2])
            else:
                if isinstance(value, integer):
                    msg = msg + '%d' % value
                else:
                    msg = msg + '%.3f' % value
        self._statusBar().showMessage(msg)

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return

        self.dimensions._update_index(event)
        if event.is_dragging:
            if self.annotation and 'Shift' in event.modifiers:
                if self._active_markers:
                    layer = self.layers[self._active_markers]
                    index = layer._selected_markers
                    if index is None:
                        pass
                    else:
                        layer.data[index] = [self.dimensions._index[1],self.dimensions._index[0],*self.dimensions._index[2:]]
                        layer._refresh()
                        self._update_statusBar()
        else:
            self._update_active_layers()
            self._update_statusBar()

    def on_mouse_press(self, event):
        if event.pos is None:
            return
        if self.annotation:
            if self._active_markers:
                layer = self.layers[self._active_markers]
                if 'Meta' in event.modifiers:
                    index = layer._selected_markers
                    if index is None:
                        pass
                    else:
                        if isinstance(layer.size, (list, ndarray)):
                            layer._size = delete(layer.size, index)
                        layer.data = delete(layer.data, index, axis=0)
                        layer._selected_markers = None
                        self._update_statusBar()
                elif 'Shift' in event.modifiers:
                    pass
                else:
                    if isinstance(layer.size, (list, ndarray)):
                        layer._size = append(layer.size, 10)
                    coord = [self.dimensions._index[1],self.dimensions._index[0],*self.dimensions._index[2:]]
                    layer.data = append(layer.data, [coord], axis=0)
                    layer._selected_markers = len(layer.data)-1
                    self._update_statusBar()

    def on_key_press(self, event):
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self.annotation:
                    self._annotation_history = True
                    self._qt.view.interactive = True
                    self.annotation = False
                    self._qt.canvas.native.setCursor(QCursor())
                else:
                    self._annotation_history = False
            elif event.key == 'Shift':
                if self.annotation and self._active_markers:
                    self._qt.canvas.native.setCursor(Qt.PointingHandCursor)
            elif event.key == 'Meta':
                if self.annotation and self._active_markers:
                    self._qt.canvas.native.setCursor(Qt.ForbiddenCursor)
            elif event.key == 'a':
                cb = self.layers._qt.layersControls.annotationCheckBox
                cb.setChecked(not cb.isChecked())

    def on_key_release(self, event):
        if event.key == ' ':
            if self._annotation_history:
                self._qt.view.interactive = False
                self.annotation = True
                if self._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self._disabled_cursor)
        elif event.key == 'Shift':
            if self.annotation:
                if self._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self._disabled_cursor)
        elif event.key == 'Meta':
                if self._active_markers:
                    self._qt.canvas.native.setCursor(Qt.CrossCursor)
                else:
                    self._qt.canvas.native.setCursor(self._disabled_cursor)
