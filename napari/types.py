from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union, NewType

import numpy as np

# This is a WOEFULLY inadqueate stub for a duck-array type.
# Mostly, just a placeholder for the concept of needing an ArrayLike type.
# It doesn't actually get used for type checking anywhere.
# Ultimately, this should come from https://github.com/napari/image-types
ArrayLike = NewType("ArrayLike", np.array)

# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type until ArrayLike is more mature.
LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]

PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]


def array_return_to_layerdata_return(
    func: Callable[[PathLike], ArrayLike]
) -> ReaderFunction:
    """Convert a PathLike -> ArrayLike function to a PathLike -> LayerData.

    Parameters
    ----------
    func : Callable[[PathLike], ArrayLike]
        A function that accepts a string or list of strings, and returns an
        ArrayLike.

    Returns
    -------
    reader_function : Callable[[PathLike], List[LayerData]]
        A function that accepts a string or list of strings, and returns data
        as a list of LayerData: List[Tuple[ArrayLike]]
    """

    @wraps(func)
    def reader_function(*args, **kwargs) -> List[LayerData]:
        result = func(*args, **kwargs)
        return [(result,)]

    return reader_function
