from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D
from napari.components.camera import DEFAULT_ORIENTATION_TYPED
from napari.settings import get_settings

if TYPE_CHECKING:
    from vispy.scene import Node
    from vispy.visuals.text.text import FontManager

    from napari.components.overlays import Overlay
    from napari.components.viewer_model import ViewerModel


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(
        self,
        *,
        viewer: ViewerModel,
        overlay: Overlay,
        parent: Node = None,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
    ):
        super().__init__(
            node=GridLines3D(
                font_manager=font_manager, font_family=font_family
            ),
            viewer=viewer,
            overlay=overlay,
            parent=parent,
        )

        self.overlay.events.color.connect(self._on_data_change)
        self.overlay.events.labels.connect(self._on_data_change)
        self.overlay.events.n_labels.connect(self._on_data_change)
        self.viewer.dims.events.order.connect(self._on_data_change)
        self.viewer.dims.events.range.connect(self._on_data_change)
        self.viewer.dims.events.ndisplay.connect(self._on_data_change)
        self.viewer.dims.events.axis_labels.connect(self._on_data_change)

        # would be nice to fire this less often to save performance
        self.viewer.camera.events.angles.connect(
            self._on_view_direction_change
        )
        self.viewer.camera.events.orientation.connect(
            self._on_view_direction_change
        )
        get_settings().appearance.events.theme.connect(self._on_data_change)
        self.viewer.events.theme.connect(self._on_data_change)

        self.reset()

    def _on_data_change(self) -> None:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        self.node.color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )
        self.node.reset_grids()
        self.node.set_extents(ranges)
        # TODO: 'force' should only be used when necesssary!
        self.node.set_ticks(
            self.overlay.labels, self.overlay.n_labels, ranges, force=True
        )
        self._on_blending_change()  # needed to ensure new grids/ticks are up to date

        self._on_view_direction_change(force=True)

    def _on_view_direction_change(self, force: bool = False) -> None:
        displayed = self.viewer.dims.displayed[::-1]
        ranges = tuple(self.viewer.dims.range[i] for i in displayed)

        if self.viewer.dims.ndisplay == 3:
            # all is flipped from zyx to xyz for vispy
            view_direction = tuple(
                np.sign(self.viewer.camera.view_direction)[::-1]
            )
            up_direction = tuple(
                np.sign(self.viewer.camera.up_direction)[::-1]
            )
            orientation_flip = tuple(
                1 if ori == default_ori else -1
                for ori, default_ori in zip(
                    self.viewer.camera.orientation,
                    DEFAULT_ORIENTATION_TYPED,
                    strict=True,
                )
            )[::-1]
        else:
            view_direction = (1, 1, 1)
            up_direction = (1, 1, 1)
            orientation_flip = (1, 1, 1)

        self.node.set_view_direction(
            ranges,
            view_direction,
            up_direction,
            orientation_flip,
            force=force,
        )

    def reset(self) -> None:
        super().reset()  # type: ignore
        self._on_data_change()
