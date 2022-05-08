from typing import Optional, Tuple, Union

from pydantic import BaseModel


class DataQueryResponse(BaseModel):
    """The result of querying layer data.

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
    value: Optional[Union[float, int]]
    position: Optional[Tuple[float, ...]]


class ShapesDataQueryResponse(DataQueryResponse):
    """A DataQueryResponse with an additional field for vertex index."""

    vertex_index: Optional[int]
