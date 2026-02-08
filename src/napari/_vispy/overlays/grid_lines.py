import numpy as np
from vispy.scene import Node

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.grid_lines import GridLines3D
from napari.components.camera import DEFAULT_ORIENTATION_TYPED
from napari.components.overlays import Overlay
from napari.components.viewer_model import ViewerModel
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.theme import get_theme


class VispyGridLinesOverlay(ViewerOverlayMixin, VispySceneOverlay):
    def __init__(
        self, *, viewer: ViewerModel, overlay: Overlay, parent: Node = None
    ):
        super().__init__(
            node=GridLines3D(),
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
        self.viewer.events.theme.connect(self._on_data_change)

        self.reset()

    def _on_data_change(self) -> None:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]

        color = self.overlay.color
        if color is None:
            # set scale color negative of theme background.
            # the reason for using the `as_hex` here is to avoid
            # `UserWarning` which is emitted when RGB values are above 1
            if (
                self.node.parent is not None
                and self.node.parent.canvas.bgcolor
            ):
                background_color = self.node.parent.canvas.bgcolor.rgba
            else:
                background_color = get_theme(self.viewer.theme).canvas.as_hex()
                background_color = transform_color(background_color)[0]
            color = np.subtract(1, background_color)
            color[-1] = background_color[-1]

        self.node.color = color
        self.node.reset_grids()
        self.node.set_extents(ranges)
        self.node.set_ticks(self.overlay.labels, self.overlay.n_labels, ranges)

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
