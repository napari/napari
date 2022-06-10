from copy import deepcopy

import napari
from napari.components.viewer_model import ViewerModel
from napari.qt import QtViewer


from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QSplitter, QPushButton, QDoubleSpinBox

class ExampleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.btn = QPushButton("Perform action")
        self.spin = QDoubleSpinBox()
        layout = QVBoxLayout()
        layout.addWidget(self.spin)
        layout.addWidget(self.btn)
        layout.addStretch(1)
        self.setLayout(layout)

class MultipleViewerWidget(QSplitter):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer_model1 = ViewerModel()
        self.viewer_model2 = ViewerModel()
        self.qt_viewer1 = QtViewer(self.viewer_model1)
        self.qt_viewer2 = QtViewer(self.viewer_model2)
        self.tab_widget = QTabWidget()
        w1 = ExampleWidget()
        w2 = ExampleWidget()
        self.tab_widget.addTab(w1, "Sample 1")
        self.tab_widget.addTab(w2, "Sample 2")
        viewer_layout = QVBoxLayout()
        viewer_layout.addWidget(self.qt_viewer1)
        viewer_layout.addWidget(self.qt_viewer2)
        w = QWidget()
        w.setLayout(viewer_layout)
        w.setContentsMargins(0, 0, 0, 0)

        self.addWidget(w)
        self.addWidget(self.tab_widget)

        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)

    def _point_update(self, event):
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = order
        order = list(self.viewer.dims.order)
        order[-3:] = order[-2], order[-1], order[-3]
        self.viewer_model2.dims.order = order

    @staticmethod
    def _clean_copy(layer):
        copy_layer = deepcopy(layer)
        copy_layer.events.scale.disconnect()
        return copy_layer

    def _layer_added(self, event):
        self.viewer_model1.layers.insert(
            event.index, self._clean_copy(event.value)
        )
        self.viewer_model2.layers.insert(
            event.index, self._clean_copy(event.value)
        )
        event.value.events.scale.connect(self._scale_sync)
        self._order_update()

    def _layer_removed(self, event):
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _layer_moved(self, event):
        print(event.index, event.new_index)
        self.viewer_model1.layers.move(event.index, event.new_index)
        self.viewer_model2.layers.move(event.index, event.new_index)\

    def _scale_sync(self, event):
        self.viewer_model1.layers[event.source.name].scale = event.source.scale
        self.viewer_model2.layers[event.source.name].scale = event.source.scale



view = napari.Viewer()
dock_widget = MultipleViewerWidget(view)

view.window.add_dock_widget(dock_widget, name="Sample")

napari.run()

