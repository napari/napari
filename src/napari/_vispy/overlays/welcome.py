from __future__ import annotations

from random import choice
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.visuals.transforms import NullTransform

from napari._vispy.overlays.base import ViewerOverlayMixin, VispyCanvasOverlay
from napari._vispy.visuals.welcome import Welcome
from napari.settings import get_settings

if TYPE_CHECKING:
    from vispy.scene import Node

    from napari import Viewer
    from napari.components.overlays import WelcomeOverlay


class VispyWelcomeOverlay(ViewerOverlayMixin, VispyCanvasOverlay):
    overlay: WelcomeOverlay
    _DEFAULT_TIP = "You're awesome!"

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

        get_settings().appearance.events.theme.connect(self._on_theme_change)

        self.overlay.events.version.connect(self._on_version_change)
        self.overlay.events.shortcuts.connect(self._on_shortcuts_change)
        self.overlay.events.tips.connect(self._on_tips_change)
        self.overlay.events.box.connect(self._on_theme_change)
        self.overlay.events.box_color.connect(self._on_theme_change)

        self.node.canvas.native.resized.connect(self._on_position_change)

        self._tips = (self._DEFAULT_TIP,)

        self.reset()

    def _should_be_visible(self) -> bool:
        return bool(super()._should_be_visible() and not self.viewer.layers)

    def _on_position_change(self, event: Any = None) -> None:
        if self.node.canvas is not None:
            x, y = np.array(self.node.canvas.size)
            self.node.set_scale_and_position(x, y)
            self.x_size = x
            self.y_size = y
        self._on_box_change()

    def _on_box_change(self) -> None:
        super()._on_box_change()
        # welcome uses some custom positioning with the transform, so copying it
        # over messes it up. We just set it to nothing.
        self.box.transform = NullTransform()
        # always opaque box color, so we hide what's behind
        bgcolor = self.box.color.rgba
        bgcolor[-1] = 1
        self.box.color = bgcolor

    def _on_theme_change(self) -> None:
        color = self._get_fgcolor()
        # dim a bit but keep opaque to avoid border artifacts
        color[:3] *= 0.7
        color[-1] = 1
        self.node.set_color(color)

    def _on_visible_change(self) -> None:
        """Update visibility and pick one tip when the overlay is shown."""
        show = self._should_be_visible()
        was_visible = self.node.visible
        self.node.visible = show
        if show and not was_visible:
            self.next_tip()

    def _on_version_change(self) -> None:
        self.node.set_version(self.overlay.version)

    def _on_shortcuts_change(self) -> None:
        self.node.set_shortcuts(self.overlay.shortcuts)

    def _on_tips_change(self) -> None:
        """Refresh available tips and update visible tip immediately."""
        self._tips = tuple(self.overlay.tips) or (self._DEFAULT_TIP,)
        if self.node.visible:
            self.next_tip()

    def next_tip(self) -> None:
        """Select and display exactly one random tip from the current tip set."""
        self.node.set_tip(choice(self._tips))

    def reset(self) -> None:
        super().reset()
        self._on_theme_change()
        self._on_version_change()
        self._on_shortcuts_change()
        self._on_tips_change()

    def close(self) -> None:
        super().close()
