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
hook implementations may accept *fewer* arguments than defined in the
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
from typing import Optional, Union, List, Any
from ..types import ReaderFunction, WriterFunction

napari_hook_specification = pluggy.HookspecMarker("napari")


# -------------------------------------------------------------------------- #
#                                 IO Hooks                                   #
# -------------------------------------------------------------------------- #


@napari_hook_specification(firstresult=True)
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]:
    """Return a function capable of loading ``path`` into napari, or ``None``.

    This is the primary "**reader plugin**" function.  It accepts a path or
    list of paths, and returns a list of data to be added to the ``Viewer``.

    The main place this hook is used is in :func:`Viewer.add_path()
    <napari.components.add_layers_mixin.AddLayersMixin.add_path>`, via the
    :func:`~napari.plugins.io.read_data_with_plugins` function.

    It will also be called on ``File -> Open...`` or when a user drops a file
    or folder onto the viewer. This function must execute *quickly*, and should
    return ``None`` if the filepath is of an unrecognized format for this
    reader plugin.  If ``path`` is determined to be recognized format, this
    function should return a *new* function that accepts the same filepath (or
    list of paths), and returns a list of ``LayerData`` tuples, where each
    tuple is a 1-, 2-, or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data,
    meta, layer_type)`` .

    ``napari`` will then use each tuple in the returned list to generate a new
    layer in the viewer using the :func:`Viewer._add_layer_from_data()
    <napari.components.add_layers_mixin.AddLayersMixin._add_layer_from_data>`
    method.  The first, (optional) second, and (optional) third items in each
    tuple in the returned layer_data list, therefore correspond to the
    ``data``, ``meta``, and ``layer_type`` arguments of the
    :func:`Viewer._add_layer_from_data()
    <napari.components.add_layers_mixin.AddLayersMixin._add_layer_from_data>`
    method, respectively.


    .. important::

       ``path`` may be either a ``str`` or a ``list`` of ``str``.  If a
       ``list``, then each path in the list can be assumed to be one part of a
       larger multi-dimensional stack (for instance: a list of 2D image files
       that should be stacked along a third axis). Implementations should do
       their own checking for ``list`` or ``str``, and handle each case as
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


@napari_hook_specification(firstresult=True)
def napari_get_writer(
    path: str, layer_types: List[str]
) -> Optional[WriterFunction]:
    """Return function capable of writing napari layer data into a `path`.

    This function will be called on File -> Save.... This function must execute
    QUICKLY, and should return ``None`` if the filepath is of an unrecognized
    format for the reader plugin or the layer types are not recognized. If the
    filepath is a recognized format, this function should return a callable
    that accepts the same filepath, a list of layer_data tuples:
    Union[Tuple[Any], Tuple[Any, Dict]], and optionally an extension.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    layer_types : list of str
        List of layer types that will be provided to the writer function.

    Returns
    -------
    Callable or None
        A function that accepts the path, a list of layer_data (where
        layer_data is (data, meta, layer_type)). If unable to write to the
        path or write the layer_data, must return ``None`` (not False).
    """


@napari_hook_specification(firstresult=True)
def napari_write_image(path: str, data: Any, meta: dict) -> bool:
    """Write image data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is from an image
        pyramid.
    meta : dict
        Image metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """


@napari_hook_specification(firstresult=True)
def napari_write_labels(path: str, data: Any, meta: dict) -> bool:
    """Write labels data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array or list of array
        Integer valued label data. Can be N dimensional. Every pixel contains
        an integer ID corresponding to the region it belongs to. The label 0 is
        rendered as transparent. If a list and arrays are decreasing in shape
        then the data is from an image pyramid.
    meta : dict
        Labels metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """


@napari_hook_specification(firstresult=True)
def napari_write_points(path: str, data: Any, meta: dict) -> bool:
    """Write points data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : array (N, D)
        Coordinates for N points in D dimensions.
    meta : dict
        Points metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """


@napari_hook_specification(firstresult=True)
def napari_write_shapes(path: str, data: Any, meta: dict) -> bool:
    """Write shapes data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : list
        List of shape data, where each element is an (N, D) array of the
        N vertices of a shape in D dimensions.
    meta : dict
        Shapes metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """


@napari_hook_specification(firstresult=True)
def napari_write_surface(path: str, data: Any, meta: dict) -> bool:
    """Write surface data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : 3-tuple of array
        The first element of the tuple is an (N, D) array of vertices of
        mesh triangles. The second is an (M, 3) array of int of indices
        of the mesh triangles. The third element is the (K0, ..., KL, N)
        array of values used to color vertices where the additional L
        dimensions are used to color the same mesh with different values.
    meta : dict
        Surface metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """


@napari_hook_specification(firstresult=True)
def napari_write_vectors(path: str, data: Any, meta: dict) -> bool:
    """Write vectors data and metadata into a path.

    Parameters
    ----------
    path : str
        Path to file, directory, or resource (like a URL).
    data : (N, 2, D) array
        The start point and projections of N vectors in D dimensions.
    meta : dict
        Vectors metadata.

    Returns
    -------
    bool : Return True if data is successfully written.
    """
