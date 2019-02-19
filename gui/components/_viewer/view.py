from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QSplitter
from PyQt5.QtGui import QCursor, QPixmap
from vispy.scene import SceneCanvas, PanZoomCamera

from os.path import join
from ...resources import resources_dir
path_cursor = join(resources_dir, 'icons', 'cursor_disabled.png')


class QtViewer(QSplitter):

    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.viewer.events.annotation.connect(self.set_annotation)
        self.viewer.events.active_markers.connect(self.set_annotation)

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(100, 100))

        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
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
        layout.addWidget(self.viewer.dims._qt)
        center.setLayout(layout)

        # Add vertical sliders, center, and layerlist
        self.addWidget(self.viewer.control_bars._qt)
        self.addWidget(center)
        self.addWidget(self.viewer.layers._qt)

        viewer.dims._qt.setFixedHeight(0)

        self._cursors = {
                'disabled': QCursor(QPixmap(path_cursor).scaled(20, 20)),
                'cross': Qt.CrossCursor,
                'forbidden': Qt.ForbiddenCursor,
                'pointing': Qt.PointingHandCursor,
                'standard': QCursor()
            }

    def set_annotation(self, event):
        if self.viewer.annotation:
            self.view.interactive = False
            if self.viewer.active_markers:
                self.canvas.native.setCursor(self._cursors['cross'])
            else:
                self.canvas.native.setCursor(self._cursors['disabled'])
        else:
            self.view.interactive = True
            self.canvas.native.setCursor(self._cursors['standard'])

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.
        """
        if event.pos is None:
            return
        self.viewer.position = event.pos
        if (event.is_dragging and self.viewer.annotation and
                'Shift' in event.modifiers and self.viewer.active_markers):
            layer = self.viewer.layers[self.viewer.active_markers]
            layer.move(self.viewer.position, self.viewer.dims.indices)
        self.viewer._update_status()

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.
        """
        if self.viewer.annotation and self.viewer.active_markers:
            layer = self.viewer.layers[self.viewer.active_markers]
            if 'Meta' in event.modifiers:
                layer.remove(self.viewer.position,
                             self.viewer.dims.indices)
            elif 'Shift' in event.modifiers:
                pass
            else:
                layer.add(self.viewer.position, self.viewer.dims.indices)
            self.viewer._update_status()

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.
        """
        return

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
                if self.viewer.annotation and self.viewer.active_markers:
                    self.canvas.native.setCursor(self._cursors['pointing'])
            elif event.key == 'Meta':
                if self.viewer.annotation and self.viewer.active_markers:
                    self.canvas.native.setCursor(self._cursors['forbidden'])
            elif event.key == 'a':
                self.viewer._set_annotation(not self.viewer.annotation)

    def on_key_release(self, event):
        if event.key == ' ':
            if self.viewer._annotation_history:
                self.view.interactive = False
                self.viewer.annotation = True
                if self.viewer.active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
        elif event.key == 'Shift':
            if self.viewer.annotation:
                if self.viewer.active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
        elif event.key == 'Meta':
                if self.viewer.active_markers:
                    self.canvas.native.setCursor(self._cursors['cross'])
                else:
                    self.canvas.native.setCursor(self._cursors['disabled'])
