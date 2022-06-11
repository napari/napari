import typing
from copy import deepcopy

from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import napari
from napari.components.layerlist import Extent
from napari.components.viewer_model import ViewerModel
from napari.layers import Vectors
from napari.qt import QtViewer

if typing.TYPE_CHECKING:
    from napari.layers import Layer


def copy_layer(layer: "Layer"):
    res_layer = deepcopy(layer)
    res_layer.events.disconnect()
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
    return res_layer


def get_property_names(layer: "Layer"):
    klass = layer.__class__
    res = []
    for event_name in layer.events.emitters:
        if event_name == "thumbnail":
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


class own_partial:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict={}):
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class CrossWidget(QCheckBox):
    def __init__(self, viewer: napari.Viewer):
        super().__init__("Add cross layer")
        self.viewer = viewer
        self.setChecked(False)
        self.stateChanged.connect(self._update_cross_visibility)
        self.layer = Vectors(name=".cross", ndim=self.viewer.dims.ndim)
        self.viewer.dims.events.order.connect(self.update_cross)
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        self.viewer.dims.events.current_step.connect(self.update_cross)

    def _update_ndim(self, event):
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
        self.layer = Vectors(name=".cross", ndim=event.value)
        self.update_cross()

    def _update_cross_visibility(self, state):
        if state:
            self.viewer.layers.append(self.layer)
        else:
            self.viewer.layers.remove(self.layer)
        self.update_cross()

    def update_cross(self):
        if self.layer not in self.viewer.layers:
            return
        # self.viewer.layers.remove(self.layer)
        extent_list = [
            layer.extent
            for layer in self.viewer.layers
            if layer is not self.layer
        ]
        extent = Extent(
            data=None,
            world=self.viewer.layers._get_extent_world(extent_list),
            step=self.viewer.layers._get_step_size(extent_list),
        )
        point = self.viewer.dims.current_step
        vec = []
        for i, (lower, upper) in enumerate(extent.world.T):
            point1 = list(point)
            point1[i] = lower + extent.step[i] / 2
            point2 = [0 for _ in point]
            point2[i] = upper - lower
            vec.append((point1, point2))
        if any(self.layer.scale != extent.step):
            self.layer.scale = extent.step
        self.layer.data = vec
        # self.viewer.layers.append(self.layer)


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
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = order

    def _layer_added(self, event):
        self.viewer_model1.layers.insert(event.index, copy_layer(event.value))
        self.viewer_model2.layers.insert(event.index, copy_layer(event.value))
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        print(get_property_names(event.value))
        self._order_update()

    def _layer_removed(self, event):
        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _layer_moved(self, event):
        print(event.index, event.new_index)
        self.viewer_model1.layers.move(event.index, event.new_index)
        self.viewer_model2.layers.move(event.index, event.new_index)

    def _property_sync(self, name, event):
        if event.source not in self.viewer.layers:
            return
        print("sync", name)
        setattr(
            self.viewer_model1.layers[event.source.name],
            name,
            getattr(event.source, name),
        )
        setattr(
            self.viewer_model2.layers[event.source.name],
            name,
            getattr(event.source, name),
        )

    def _scale_sync(self, event):
        self.viewer_model1.layers[event.source.name].scale = event.source.scale
        self.viewer_model2.layers[event.source.name].scale = event.source.scale


view = napari.Viewer()
dock_widget = MultipleViewerWidget(view)
cross = CrossWidget(view)

view.window.add_dock_widget(dock_widget, name="Sample")
view.window.add_dock_widget(cross, name="Cross", area="left")

napari.run()
