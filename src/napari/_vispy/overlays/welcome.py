from __future__ import annotations

from itertools import cycle
from random import sample
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.app.timer import Timer

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.welcome import Welcome

if TYPE_CHECKING:
    from vispy.scene import Node
    from vispy.util.event import Event

    from napari import Viewer
    from napari.components.overlays import WelcomeOverlay


class VispyWelcomeOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    overlay: WelcomeOverlay

    def __init__(
        self,
        *,
        viewer: Viewer,
        overlay: WelcomeOverlay,
        parent: Node | None = None,
    ) -> None:
        super().__init__(
            node=Welcome(), viewer=viewer, overlay=overlay, parent=parent
        )
        self.viewer.events.theme.connect(self._on_theme_change)
        self.viewer.layers.events.inserted.connect(self._on_visible_change)
        self.viewer.layers.events.removed.connect(self._on_visible_change)

        self.overlay.events.version.connect(self._on_version_change)
        self.overlay.events.shortcuts.connect(self._on_shortcuts_change)
        self.overlay.events.tips.connect(self._on_tips_change)
        self.overlay.events.box.connect(self._on_theme_change)
        self.overlay.events.box_color.connect(self._on_theme_change)

        self.node.canvas.native.resized.connect(self._on_position_change)

        self.tips_iterator = cycle(["You're awesome!"])
        self.tip_timer = Timer(10, self.next_tip)
        self.next_tip()

        self.reset()

    def _on_position_change(self, event: Any = None) -> None:
        if self.node.canvas is not None:
            x, y = np.array(self.node.canvas.size)
            self.node.set_scale_and_position(x, y)
            self.x_size = x
            self.y_size = y

    def _on_box_change(self) -> None:
        super()._on_box_change()
        # welcome uses some custom positioning with the transform
        self.box.center = (0, 0)

    def _on_theme_change(self) -> None:
        color = self._get_fgcolor()
        color[-1] *= 0.7  # dim a bit
        self.node.set_color(color)

    def _on_visible_change(self) -> None:
        show = self.overlay.visible and not self.viewer.layers
        self.node.visible = show
        if show:
            self.tip_timer.start()
        else:
            try:
                self.tip_timer.stop()
            except RuntimeError as e:  # pragma: no cover
                if (
                    'wrapped C/C++ object of type' not in e.args[0]
                    and 'Internal C++ object' not in e.args[0]
                ):
                    # checking if the object is partially deleted. Otherwise
                    # reraise exception. For more details see:
                    # https://github.com/napari/napari/pull/5499
                    raise

    def _on_version_change(self) -> None:
        self.node.set_version(self.overlay.version)

    def _on_shortcuts_change(self) -> None:
        self.node.set_shortcuts(self.overlay.shortcuts)

    def _on_tips_change(self) -> None:
        if self.overlay.tips:
            self.tips_iterator = cycle(
                sample(self.overlay.tips, len(self.overlay.tips))
            )
        else:
            self.tips_iterator = cycle(["You're awesome!"])

    def next_tip(self, event: Event | None = None) -> None:
        self.node.set_tip(next(self.tips_iterator))

    def reset(self) -> None:
        super().reset()
        self._on_theme_change()
        self._on_version_change()
        self._on_shortcuts_change()
        self._on_tips_change()
        self.next_tip()

    def close(self) -> None:
        self.tip_timer.stop()
        super().close()
