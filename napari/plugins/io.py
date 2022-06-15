from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, cast

from ..utils.misc import abspath_or_url
from ..utils.translations import trans
from . import _npe2

if TYPE_CHECKING:
    from npe2.manifest.contributions import WriterContribution

    from ..layers import Layer
    from ..types import LayerData


def read_data_with_plugins(
    paths: Sequence[str],
    plugin: Optional[str] = None,
    stack: bool = False,
) -> Tuple[List[LayerData], str]:
    """Iterate reader hooks and return first non-None LayerData or None.

    This function returns as soon as the path has been read successfully,
    while catching any plugin exceptions, storing them for later retrieval,
    providing useful error messages, and re-looping until either a read
    operation was successful, or no valid readers were found.

    Exceptions will be caught and stored as PluginErrors
    (in plugins.exceptions.PLUGIN_ERRORS)

    Parameters
    ----------
    paths : str, or list of string
        The of path (file, directory, url) to open
    plugin : str, optional
        Name of a plugin to use.  If provided, will force ``path`` to be read
        with the specified ``plugin``.  If the requested plugin cannot read
        ``path``, a PluginCallError will be raised.
    stack : bool
        See `Viewer.open`

    Returns
    -------
    LayerData : list of tuples, or None
        LayerData that can be passed to :func:`Viewer._add_layer_from_data()
        <napari.components.viewer_model.ViewerModel._add_layer_from_data>`.
        ``LayerData`` is a list tuples, where each tuple is one of
        ``(data,)``, ``(data, meta)``, or ``(data, meta, layer_type)`` .

        If no reader plugins were found (or they all failed), returns ``None``

    Raises
    ------
    PluginCallError
        If ``plugin`` is specified but raises an Exception while reading.
    """
    if plugin == 'builtins':
        warnings.warn(
            'The "builtins" plugin name is deprecated and will not work in a future '
            'version. Please use "napari" instead.',
        )
        plugin = 'napari'

    assert isinstance(paths, list)
    if not stack:
        assert len(paths) == 1

    paths = [abspath_or_url(p, must_exist=True) for p in paths]
    if (res := _npe2.read(paths, plugin, stack=stack)) is not None:
        _ld, plugin_name = res
        data = [] if _is_null_layer_sentinel(_ld) else _ld
        return (cast(List['LayerData'], data), plugin_name)

    if plugin:
        message = trans._(
            'Plugin {plugin!r} not capable of reading {repr_path!r}.',
            deferred=True,
            plugin=plugin,
            repr_path=paths,
        )
    # if layer_data is empty, it means no plugin could read path
    # we just want to provide some useful feedback, which includes
    # whether or not paths were passed to plugins as a list.
    elif stack:
        message = trans._(
            'No plugin found capable of reading [{repr_path!r}, ...] as stack.',
            deferred=True,
            repr_path=paths[0],
        )
    else:
        message = trans._(
            'No plugin found capable of reading {repr_path!r}.',
            deferred=True,
            repr_path=paths,
        )

    raise ValueError(message)


def save_layers(
    path: str,
    layers: List[Layer],
    *,
    plugin: Optional[str] = None,
    _writer: Optional[WriterContribution] = None,
) -> List[str]:
    """Write list of layers or individual layer to a path using writer plugins.

    Parameters
    ----------
    path : str
        A filepath, directory, or URL to open.
    layers : List[layers.Layer]
        List of layers to be saved. If only a single layer is passed then
        we use the hook specification corresponding to its layer type,
        ``napari_write_<layer_type>``. If multiple layers are passed then we
        use the ``napari_get_writer`` hook specification.
    plugin : str, optional
        Name of the plugin to use for saving. If None then all plugins
        corresponding to appropriate hook specification will be looped
        through to find the first one that can save the data.

    Returns
    -------
    list of str
        File paths of any files that were written.
    """
    if len(layers) > 1:
        written = _write_multiple_layers_with_plugins(
            path, layers, plugin_name=plugin, _writer=_writer
        )
    elif len(layers) == 1:
        _written = _write_single_layer_with_plugins(
            path, layers[0], plugin_name=plugin, _writer=_writer
        )
        written = [_written] if _written else []
    else:
        written = []

    if not written:
        # if written is empty, it means no plugin could write the
        # path/layers combination
        # we just want to provide some useful feedback
        warnings.warn(
            trans._(
                'No data written! There may be no plugins capable of writing these {length} layers to {path}.',
                deferred=True,
                length=len(layers),
                path=path,
            )
        )

    return written


def _is_null_layer_sentinel(layer_data: Any) -> bool:
    """Checks if the layer data returned from a reader function indicates an
    empty file. The sentinel value used for this is ``[(None,)]``.

    Parameters
    ----------
    layer_data : LayerData
        The layer data returned from a reader function to check

    Returns
    -------
    bool
        True, if the layer_data indicates an empty file, False otherwise
    """
    return (
        isinstance(layer_data, list)
        and len(layer_data) == 1
        and isinstance(layer_data[0], tuple)
        and len(layer_data[0]) == 1
        and layer_data[0][0] is None
    )


def _write_multiple_layers_with_plugins(
    path: str,
    layers: List[Layer],
    *,
    plugin_name: Optional[str] = None,
    _writer: Optional[WriterContribution] = None,
) -> List[str]:
    """Write data from multiple layers data with a plugin.

    If a ``plugin_name`` is not provided we loop through plugins to find the
    first one that knows how to handle the combination of layers and is able to
    write the file. If no plugins offer ``napari_get_writer`` for that
    combination of layers then the default ``napari_get_writer`` will create a
    folder and call ``napari_write_<layer>`` for each layer using the
    ``layer.name`` variable to modify the path such that the layers are written
    to unique files in the folder.

    If a ``plugin_name`` is provided, then call ``napari_get_writer`` for that
    plugin. If it doesn't return a ``WriterFunction`` we error, otherwise we
    call it and if that fails if it we error.

    Exceptions will be caught and stored as PluginErrors
    (in plugins.exceptions.PLUGIN_ERRORS)

    Parameters
    ----------
    path : str
        The path (file, directory, url) to write.
    layers : List of napari.layers.Layer
        List of napari layers to write.
    plugin_name : str, optional
        If provided, force the plugin manager to use the ``napari_get_writer``
        from the requested ``plugin_name``.  If none is available, or if it is
        incapable of handling the layers, this function will fail.

    Returns
    -------
    list of str
        A list of filenames, if any, that were written.
    """
    if written_paths := _npe2.write_layers(path, layers, plugin_name, _writer):
        return written_paths

    layer_types = [layer._type_string for layer in layers]

    if plugin_name:
        msg = trans._(
            'Requested plugin "{plugin_name}" is not capable of writing this combination of layer types: {layer_types}',
            deferred=True,
            plugin_name=plugin_name,
            layer_types=layer_types,
        )
    else:
        msg = trans._(
            'Unable to find plugin capable of writing this combination of layer types: {layer_types}',
            deferred=True,
            layer_types=layer_types,
        )

    raise ValueError(msg)


def _write_single_layer_with_plugins(
    path: str,
    layer: Layer,
    *,
    plugin_name: Optional[str] = None,
    _writer: Optional[WriterContribution] = None,
) -> Optional[str]:
    """Write single layer data with a plugin.

    Parameters
    ----------
    path : str
        The path (file, directory, url) to write.
    layer : napari.layers.Layer
        Layer to be written out.
    plugin_name : str, optional
        Name of the plugin to write data with. If None then all plugins
        corresponding to appropriate hook specification will be looped
        through to find the first one that can write the data.

    Returns
    -------
    path : str or None
        If data is successfully written, return the ``path`` that was written.
        Otherwise, if nothing was done, return ``None``.
    """

    if written_paths := _npe2.write_layers(
        path, [layer], plugin_name, _writer
    ):
        return written_paths[0]

    raise ValueError(
        trans._(
            "No plugin capable of writing {layer_type} layer",
            deferred=True,
            layer_type=layer._type_string,
        )
    )
