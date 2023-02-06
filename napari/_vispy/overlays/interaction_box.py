from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.layers.base._base_constants import InteractionBoxHandle


class _VispyBoundingBoxOverlay(LayerOverlayMixin, VispySceneOverlay):
    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            node=InteractionBox(),
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.layer.events.set_data.connect(self._on_visible_change)

    def _on_bounds_change(self):
        pass

    def _on_visible_change(self):
        if self.layer._slice_input.ndisplay == 2:
            super()._on_visible_change()
            self._on_bounds_change()
        else:
            self.node.visible = False

    def reset(self):
        super().reset()
        self._on_bounds_change()


class VispySelectionBoxOverlay(_VispyBoundingBoxOverlay):
    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.overlay.events.bounds.connect(self._on_bounds_change)
        self.overlay.events.handles.connect(self._on_bounds_change)
        self.overlay.events.selected_handle.connect(self._on_bounds_change)

    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            top_left, bot_right = self.overlay.bounds
            self.node.set_data(
                # invert axes for vispy
                top_left[::-1],
                bot_right[::-1],
                handles=self.overlay.handles,
                selected=self.overlay.selected_handle,
            )


class VispyTransformBoxOverlay(_VispyBoundingBoxOverlay):
    def __init__(self, *, layer, overlay, parent=None) -> None:
        super().__init__(
            layer=layer,
            overlay=overlay,
            parent=parent,
        )
        self.layer.events.scale.connect(self._on_bounds_change)
        self.layer.events.translate.connect(self._on_bounds_change)
        self.layer.events.rotate.connect(self._on_bounds_change)
        self.layer.events.shear.connect(self._on_bounds_change)
        self.layer.events.affine.connect(self._on_bounds_change)
        self.overlay.events.selected_handle.connect(self._on_bounds_change)

    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            bounds = self.layer._display_bounding_box(
                self.layer._slice_input.displayed
            )
            # invert axes for vispy
            top_left, bot_right = (tuple(point) for point in bounds.T[:, ::-1])

            if self.overlay.selected_handle == InteractionBoxHandle.INSIDE:
                selected = slice(None)
            else:
                selected = self.overlay.selected_handle

            self.node.set_data(
                top_left,
                bot_right,
                handles=True,
                selected=selected,
            )
