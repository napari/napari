from typing import Union

from ...utils.events import EventedModel
from ...utils.events.custom_types import Array


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
    color: Union[str, Array[float, (3,)], Array[float, (4,)]] = 'black'
    width: float = 1
