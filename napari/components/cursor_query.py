from typing import Optional, Tuple, Union

from pydantic import BaseModel


class CursorQuery(BaseModel):
    """The result of querying layer data under the cursor.

    Attributes
    ----------
    index: int
        Index of the first visible object under the cursor.
    value: int | float
        Value of data at the ray-data intersection.
    intersection: tuple of float
        ray-data intersection.
    """

    index: Optional[int]
    value: Optional[Union[int, float]]
    intersection: Optional[Tuple[float, ...]]
