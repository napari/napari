from __future__ import annotations

from napari.components.overlays.base import SceneOverlay


class LabelsBrushStrokeOverlay(SceneOverlay):
    """Overlay for the right-click "encircle and fill" brush stroke.

    Attributes
    ----------
    box : bool
        Whether the background box is visible or not.
    box_color : ColorValue or None
        Background box color. If unset, it defaults to the canvas color.
    gridded : bool
        The overlay will be duplicated across all grid cells in gridded mode.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    blending : Blending
        One of a list of preset blending modes that determines how RGB and
        alpha values of the overlay get mixed with the visuals below.
    """

    position: tuple[float, ...] | None = None
