from enum import Enum, auto

from pydantic import Field

from ...utils.color import ColorValue
from ...utils.events import EventedModel


class NormalMode(Enum):
    FACE = auto()
    VERTEX = auto()


class Normals(EventedModel):
    """
    Represents face or vertex normals of a surface mesh.

    Attributes
    ----------
    mode: str
        Which normals to display (face or vertex). Immutable Field.
    visible : bool
        Whether the normals are displayed.
    color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements.
    width : float
        The width of the normal lines.
    length : float
        The length of the face normal lines.
    """

    mode: NormalMode = Field(NormalMode.FACE, allow_mutation=False)
    visible: bool = False
    color: ColorValue
    width: float = 1
    length: float = 5


class SurfaceNormals(EventedModel):
    """
    Represents both face and vertex normals for a surface mesh.
    """

    face: Normals = Field(Normals(mode=NormalMode.FACE, color='orange'))
    vertex: Normals = Field(Normals(mode=NormalMode.FACE, color='blue'))
