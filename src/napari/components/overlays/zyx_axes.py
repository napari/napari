from napari.components.overlays.base import CanvasOverlay


class ZYXAxesOverlay(CanvasOverlay):
    """Axes indicating camera orientation, pinned to a canvas corner.

    Attributes
    ----------
    labels : bool
        If axis labels ('Z', 'Y', 'X') are visible or not.
    colored : bool
        If axes are colored or not. If colored then default
        coloring is z=magenta, y=yellow, x=cyan. If not
        colored than axes are the color opposite of
        the canvas background.
    dashed : bool
        If axes are dashed or not. If not dashed then
        all the axes are solid. If dashed then x=solid,
        y=dashed, z=dotted.
    arrows : bool
        If axes have arrowheads or not.
    visible : bool
        If the overlay is visible or not.
    opacity : float
        The opacity of the overlay. 0 is fully transparent.
    order : int
        The rendering order of the overlay: lower numbers get rendered first.
    position : str
        The position of the overlay in the canvas. Must be one of
        'top_left', 'top_right', 'bottom_left', 'bottom_right',
        'top_center', 'bottom_center'.
    """

    labels: bool = True
    colored: bool = True
    dashed: bool = False
    arrows: bool = True
