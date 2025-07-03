from napari.components.overlays.base import SceneOverlay


class AxesOverlay(SceneOverlay):
    """Axes indicating world coordinate origin and orientation.

    Attributes
    ----------
    labels : bool
        If axis labels are visible or not. Note that the actual
        axis labels are stored in `viewer.dims.axis_labels`.
    colored : bool
        If axes are colored or not. If colored then default
        coloring is x=cyan, y=yellow, z=magenta. If not
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
    """

    labels: bool = True
    colored: bool = True
    dashed: bool = False
    arrows: bool = True
