"""
All napari hook specifications for pluggable functionality are defined here.

A *hook specification* is a function signature (with documentation) that
declares an API that plugin developers must adhere to when providing hook
implementations.  *Hook implementations* provided by plugins (and internally by
napari) will then be invoked in various places throughout the code base.

When implementing a hook specification, pay particular attention to the number
and types of the arguments in the specification signature, as well as the
expected return type.

To allow for hook specifications to evolve over the lifetime of napari,
hook implementations may accept *less* arguments then defined in the
specification. (This allows for extending existing hook arguments without
breaking existing implementations). However, implementations must not require
*more* arguments than defined in the spec.

Hook specifications are a feature of
`pluggy <https://pluggy.readthedocs.io/en/latest/#specs>`_.

.. NOTE::
    in the `pluggy documentation <https://pluggy.readthedocs.io/en/latest/>`_,
    hook specification marker instances are named ``hookspec`` by convention,
    and hook implementation marker instances are named ``hookimpl``.  The
    convention in napari is to name them more explicity:
    ``napari_hook_specification`` and ``napari_hook_implementation``,
    respectively.
"""

# These hook specifications also serve as the API reference for plugin
# developers, so comprehensive documentation with complete type annotations is
# imperative!

import pluggy
from typing import Optional, Union, List
from ..types import ReaderFunction

napari_hook_specification = pluggy.HookspecMarker("napari")


# -------------------------------------------------------------------------- #
#                                 IO Hooks                                   #
# -------------------------------------------------------------------------- #


@napari_hook_specification(firstresult=True)
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]:
    """Return a function capable of loading ``path`` into napari, or ``None``.

    This is the primary **reader plugin** function.

    It will be called on ``File -> Open...`` or when a user drops a file or
    folder onto the viewer. This function must execute *quickly*, and should
    return ``None`` if the filepath is of an unrecognized format for this
    reader plugin.  If the filepath is a recognized format, this function
    should return a callable that accepts the same filepath, and returns a list
    of layer_data tuples: ``Union[Tuple[Any], Tuple[Any, Dict]]``.

    The main place this hook is used is in :func:`Viewer.add_path()
    <napari.components.add_layers_mixin.AddLayersMixin.add_path>`, via the
    :func:`~napari.plugins.utils.get_layer_data_from_plugins` function.

    Note: ``path`` may be either a str or a list of str, and implementations
    should do their own checking for list or str, and handle each case as
    desired.

    Parameters
    ----------
    path : str or list of str
        Path to file, directory, or resource (like a URL), or a list of paths.

    Returns
    -------
    Callable or None
        A function that accepts the path, and returns a list of ``layer_data``,
        where ``layer_data`` is one of ``(data,)``, ``(data, meta)``, or
        ``(data, meta, layer_type)``.
        If unable to read the path, must return ``None`` (not ``False``!).
    """
