import pluggy
from typing import Callable, Optional, List, Tuple, Union, Any, Dict

hookspec = pluggy.HookspecMarker("napari")

# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type for now
LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
ReaderFunction = Callable[[str], List[LayerData]]


@hookspec(firstresult=True)
def napari_get_reader(path: str) -> Optional[ReaderFunction]:
    """Return function capable of loading `path` into napari, or None.

    This function will be called on File -> Open... or when a user drags and
    drops a file/folder onto the viewer. This function must execute QUICKLY,
    and should return ``None`` if the filepath is of an unrecognized format
    for this reader plugin.  If the filepath is a recognized format, this
    function should return a callable that accepts the same filepath, and
    returns a list of layer_data tuples: Union[Tuple[Any], Tuple[Any, Dict]].

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL)

    Returns
    -------
    {function, None}
        A function that accepts the path, and returns a list of layer_data
        (where layer_data is one of (data,), (data, meta), or
        (data, meta, layer_type)).
        If unable to read the path, must return ``None`` (not False).
    """
