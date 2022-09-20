from ...utils.color import ColorValue
from ...utils.events import EventedModel


class SurfaceWireframe(EventedModel):
    """
    Wireframe representation of the edges of a surface mesh.

    Attributes
    ----------
    visible : bool
        Whether the wireframe is displayed.
    color : ColorValue
        The color of the wireframe lines.
        See ``ColorValue.validate`` for supported values.
    width : float
        The width of the wireframe lines.
    """

    visible: bool = False
    color: ColorValue = 'black'
    width: float = 1
