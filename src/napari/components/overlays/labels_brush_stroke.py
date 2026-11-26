from __future__ import annotations

from typing import TYPE_CHECKING

from napari.components.overlays.base import SceneOverlay

if TYPE_CHECKING:
    from napari.layers import Labels


class LabelsBrushStrokeOverlay(SceneOverlay):
    """Overlay for the right-click "encircle and fill" brush stroke.

    While a stroke is active, a circle of `radius` (in data units) is drawn
    centered on `center` to indicate the radius within which returning the
    cursor completes the stroke.

    Attributes
    ----------
    enabled : bool
        Whether the overlay is active (gated to PAINT mode).
    active : bool
        Whether a stroke is currently in progress (circle visible).
    center : tuple
        Data coordinates (full-ndim) of the initial right-click point.
    radius : float
        Radius of the stop circle, in data units.
    """

    enabled: bool = False
    active: bool = False
    center: tuple = (0.0, 0.0)
    radius: float = 0.0

    def abort(self, layer: Labels) -> None:
        """Revert the staged (uncommitted) stroke pixels and end the stroke."""
        layer._abort_stroke()
        self.active = False
