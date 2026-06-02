from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.interaction_box import InteractionBox
from napari.layers.base._base_constants import InteractionBoxHandle
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.components.overlays import (
        SceneOverlay,
        SelectionBoxOverlay,
        TransformBoxOverlay,
    )


class _VispyBoundingBoxOverlay(LayerOverlayMixin, VispySceneOverlay):
    node: InteractionBox
    overlay: SceneOverlay

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(node=InteractionBox(), **kwargs)
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
    overlay: SelectionBoxOverlay

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.overlay.events.bounds.connect(self._on_bounds_change)
        self.overlay.events.handles.connect(self._on_bounds_change)
        self.overlay.events.selected_handle.connect(self._on_bounds_change)
        get_settings().appearance.highlight.events.connect(
            self._on_settings_change
        )

        self.reset()

    def _on_settings_change(self):
        self.node.highlight_color = (
            get_settings().appearance.highlight.highlight_color
        )
        self.node.highlight_width = (
            get_settings().appearance.highlight.highlight_thickness
        )
        self._on_bounds_change()

    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            bounds = np.array(self.overlay.bounds)
            top_left, bot_right = (tuple(point) for point in bounds[:, ::-1])

            if self.overlay.selected_handle == InteractionBoxHandle.INSIDE:
                selected = slice(None)
            else:
                selected = self.overlay.selected_handle

            self.node.set_data(
                top_left,
                bot_right,
                handles=self.overlay.handles,
                selected=selected,
            )

    def reset(self):
        super().reset()
        self._on_settings_change()


class VispyTransformBoxOverlay(_VispyBoundingBoxOverlay):
    overlay: TransformBoxOverlay

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer.events.scale.connect(self._on_bounds_change)
        self.layer.events.translate.connect(self._on_bounds_change)
        self.layer.events.rotate.connect(self._on_bounds_change)
        self.layer.events.shear.connect(self._on_bounds_change)
        self.layer.events.affine.connect(self._on_bounds_change)
        self.overlay.events.selected_handle.connect(self._on_bounds_change)

        self.reset()

    def _on_bounds_change(self):
        if self.layer._slice_input.ndisplay == 2:
            bounds = self.layer._display_bounding_box_augmented_data_level(
                self.layer._slice_input.displayed
            )
            # invert axes for vispy
            top_left, bot_right = (tuple(point) for point in bounds.T[:, ::-1])
            selected: int | slice | None

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
