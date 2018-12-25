from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QSplitter
from PyQt5.QtGui import QCursor, QPixmap
from vispy.scene import SceneCanvas, PanZoomCamera

from numpy import clip, integer, ndarray, append, insert, delete, empty
from copy import copy

from os.path import join
from ...icons import icons_dir
path_cursor = join(icons_dir, 'cursor_disabled.png')

class QtViewer(QSplitter):

    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(100, 100))

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_key_release)

        self.view = self.canvas.central_widget.add_view()
        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()

        center = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        layout.addWidget(self.viewer.dimensions._qt)
        center.setLayout(layout)

        # Add vertical sliders, center, and layerlist
        self.addWidget(self.viewer.control_bars._qt)
        self.addWidget(center)
        self.addWidget(self.viewer.layers._qt)

        viewer.dimensions._qt.setFixedHeight(0)

        self._cursors = {
            'diabled' : QCursor(QPixmap(path_cursor).scaled(20,20)),
            'cross' : Qt.CrossCursor,
            'forbidden' : Qt.ForbiddenCursor,
            'pointing' : Qt.PointingHandCursor,
            'standard' : QCursor()
        }


    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if self.viewer.layers:
            if event.pos is None:
                return

            self.viewer.dimensions._update_index(event)
            if event.is_dragging:
                if self.viewer.annotation and 'Shift' in event.modifiers:
                    if self.viewer._active_markers:
                        layer = self.viewer.layers[self.viewer._active_markers]
                        index = layer._selected_markers
                        if index is None:
                            pass
                        else:
                            layer.data[index] = [self.viewer.dimensions._index[1],self.viewer.dimensions._index[0],*self.viewer.dimensions._index[2:]]
                            layer._refresh()
                            self.viewer._update_status_bar()
            else:
                self.viewer._update_active_layers(None)
                self.viewer._update_status_bar()

    def on_mouse_press(self, event):
        if self.viewer.layers:
            if event.pos is None:
                return
            if self.viewer.annotation:
                if self.viewer._active_markers:
                    layer = self.viewer.layers[self.viewer._active_markers]
                    if 'Meta' in event.modifiers:
                        index = layer._selected_markers
                        if index is None:
                            pass
                        else:
                            if isinstance(layer.size, (list, ndarray)):
                                layer._size = delete(layer.size, index)
                            layer.data = delete(layer.data, index, axis=0)
                            layer._selected_markers = None
                            self.viewer._update_status_bar()
                    elif 'Shift' in event.modifiers:
                        pass
                    else:
                        if isinstance(layer.size, (list, ndarray)):
                            layer._size = append(layer.size, 10)
                        coord = [self.viewer.dimensions._index[1],self.viewer.dimensions._index[0],*self.viewer.dimensions._index[2:]]
                        layer.data = append(layer.data, [coord], axis=0)
                        layer._selected_markers = len(layer.data)-1
                        self.viewer._update_status_bar()

    def on_key_press(self, event):
        if event.native.isAutoRepeat():
            return
        else:
            if event.key == ' ':
                if self.viewer.annotation:
                    self.viewer._annotation_history = True
                    self.view.interactive = True
                    self.viewer.annotation = False
                    self.canvas.native.setCursor(self._cursors['standard'])
                else:
                    self.viewer._annotation_history = False
            elif event.key == 'Shift':
                if self.viewer.annotation and self.viewer._active_markers:
                    self.canvas.native.setCursor(self._cursors['pointing'])
            elif event.key == 'Meta':
                if self.viewer.annotation and self.viewer._active_markers:
                    self.canvas.native.setCursor(self._cursors['forbidden'])
            elif event.key == 'a':
                cb = self.viewer.layers._qt.layerButtons.annotationCheckBox
                cb.setChecked(not cb.isChecked())

    def on_key_release(self, event):
        if event.key == ' ':
            if self.viewer._annotation_history:
                self.view.interactive = False
                self.viewer.annotation = True
                if self.viewer._active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
        elif event.key == 'Shift':
            if self.viewer.annotation:
                if self.viewer._active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
        elif event.key == 'Meta':
                if self.viewer._active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
