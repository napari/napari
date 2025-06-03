import numpy as np
from qtpy.QtWidgets import QWidget
from vispy.scene import Box, Line, PanZoomCamera, SceneCanvas
from vispy.visuals.transforms import MatrixTransform

import napari


class ClippingPlanesControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._active_layer = None
        self.viewer = viewer

        self.canvas = SceneCanvas(
            keys=None, size=(600, 600), vsync=True, parent=self
        )
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = PanZoomCamera(aspect=1)
        self.view.camera.interactive = False

        self.box = Box(
            10,
            20,
            30,
            face_colors=np.repeat([[0, 1, 1]], 12, axis=0),
            edge_color='black',
            parent=self.view.scene,
        )
        self.box.transform = MatrixTransform()
        self.box.transform.rotate(23, (1, 1, 1))

        self.view.camera.set_range(margin=0.5)

        pos = np.array([(0, -100, -200), (0, 100, -200)])

        self.left = Line(pos, color='red', parent=self.view.scene)
        self.right = Line(pos, color='red', parent=self.view.scene)

        self.viewer.dims.events.range.connect(self._on_extent_change)

        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_change
        )

    def _on_active_layer_change(self):
        old_layer = self._active_layer
        self._active_layer = self.viewer.layers.selection.active

        if old_layer is not None:
            old_layer.experimental_clipping_planes.events.disconnect(
                self._on_planes_change
            )

        self._active_layer.experimental_clipping_planes.events.connect(
            self._on_planes_change
        )
        if hasattr(self._active_layer, 'selected_label'):
            selection_event = self._active_layer.events.selected_label
        elif hasattr(self._active_layer, 'selected_data'):
            selection_event = self._active_layer.selected_data.events
        selection_event.connect(self._on_layer_selection_changed)

        self._on_layer_selection_changed()
        self._on_features_change()
        self.toggle.setVisible(True)
        self.save.setVisible(True)
        self.table.setVisible(True)

        if self._active_layer is None:
            self.info.setText('No layer selected.')
        elif not hasattr(self._active_layer, 'features'):
            self.info.setText(
                f'"{self._active_layer.name}" has no features table.'
            )
        else:
            self.info.setText(f'Features of "{self._active_layer.name}"')

    def _on_extent_change(self):
        _ = [self.viewer.dims.range[i] for i in self.viewer.dims.displayed]
