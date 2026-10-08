from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vispy.scene.visuals import Ellipse

from napari._vispy.overlays.base import LayerOverlayMixin, VispySceneOverlay
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.components.overlays import LabelsBrushStrokeOverlay
    from napari.layers import Labels


class VispyLabelsBrushStrokeOverlay(LayerOverlayMixin, VispySceneOverlay):
    layer: Labels
    overlay: LabelsBrushStrokeOverlay

    def __init__(self, **kwargs):
        self._circle = Ellipse(
            center=(0, 0),
            radius=1.0,
            color=(0, 0, 0, 0),  # transparent fill
            border_color=(1, 0, 0, 1),  # red outline
            border_method='agg',
        )

        super().__init__(node=self._circle, **kwargs)

        self.overlay.events.position.connect(self._on_position_change)

        adv_settings = get_settings().advanced
        self._radius = adv_settings.paint_fill_completion_radius
        adv_settings.events.paint_fill_completion_radius.connect(
            self._on_radius_change
        )

        self.reset()

    def _on_radius_change(self, value):
        self._radius = value

    def _on_position_change(self, event=None):
        if self.overlay.position is None:
            self._circle.visible = False
            return

        dd = list(self.viewer.dims.displayed)
        center = np.array(self.overlay.position)

        # convert data -> texture space when downsampling is active, like the
        # labels polygon overlay does for its points
        radius = self.layer.brush_size * self._radius
        tile2data = self.layer._transforms['tile2data']
        if hasattr(tile2data, 'scale') and not np.allclose(
            tile2data.scale, 1.0
        ):
            center = np.asarray(tile2data.inverse(center), dtype=float)
            scale = np.abs(np.asarray(tile2data.scale))[dd]
            radius = (radius / scale[1], radius / scale[0])

        self._circle.center = tuple(center[dd][::-1])
        self._circle.radius = radius
        self._circle.visible = True

    def reset(self):
        super().reset()
        self._on_position_change()
