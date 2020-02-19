"""
All hook specifications for pluggable functionality should be defined here.

A hook specification is a function signature (with documentation) that declares
an API that plugin developers must adhere to when providing hook
implementations.  Hook implementations provided by plugins (and internally by
napari) will then be invoked in various places throughout the code base.

NOTE: in pluggy documentation, hook specification marker instances are named
"hookspec" by convention, and hook implementation marker instances are named
"hookimpl".  The convention in napari is to name them more explicity:
"napari_hook_specification" and "napari_hook_implementation", respectively.

hook specifications are a feature of pluggy:
https://pluggy.readthedocs.io/en/latest/#specs

These hook specifications also serve as the API reference for plugin
developers, so comprehensive documentation with complete type annotations is
imperative.
"""

import pluggy
from typing import Callable, Optional, List, Tuple, Union, Any, Dict

napari_hook_specification = pluggy.HookspecMarker("napari")

# layer data may be: (data,) (data, meta), or (data, meta, layer_type)
# using "Any" for the data type for now
LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
ReaderFunction = Callable[[str], List[LayerData]]


@napari_hook_specification(firstresult=True)
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
    Callable or None
        A function that accepts the path, and returns a list of layer_data
        (where layer_data is one of (data,), (data, meta), or
        (data, meta, layer_type)).
        If unable to read the path, must return ``None`` (not False).
    """
