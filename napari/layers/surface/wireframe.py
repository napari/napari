from napari.utils.color import ColorValue

from ...utils.events import EventedModel


class SurfaceWireframe(EventedModel):
    """
    Wireframe representation of the edges of a surface mesh.

    Attributes
    ----------
    visible : bool
        Whether the wireframe is displayed.
    color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements.
    width : float
        The width of the wireframe lines.
    """

    visible: bool = False
    # TODO: check if this change in stored value is desired.
    color: ColorValue = 'black'
    width: float = 1
