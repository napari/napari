from typing import Optional, Tuple, Union

from pydantic import BaseModel


class CursorQuery(BaseModel):
    """The result of querying layer data under the cursor.

    Attributes
    ----------
    index: int
        Index of the first visible object under the cursor.
    value: int | float
        Value of data sampled at a cursor position.
    position: tuple of float
        2D: position of cursor in data coordinates.
        3D: position of relevant ray-data intersection in data coordinates.
    """

    index: Optional[int]
    value: Optional[Union[int, float]]
    position: Optional[Tuple[float, ...]]
