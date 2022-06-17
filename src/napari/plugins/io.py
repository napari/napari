from __future__ import annotations

import os
import pathlib
import warnings
from logging import getLogger
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from napari_plugin_engine import HookImplementation, PluginCallError

from ..layers import Layer
from ..types import LayerData
from ..utils.misc import abspath_or_url
from ..utils.translations import trans
from . import _npe2, plugin_manager

logger = getLogger(__name__)
if TYPE_CHECKING:
    from npe2.manifest.contributions import WriterContribution


def read_data_with_plugins(
    paths: Sequence[str],
    plugin: Optional[str] = None,
    stack: bool = False,
) -> Tuple[Optional[List[LayerData]], Optional[HookImplementation]]:
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
    hookimpl: Optional[HookImplementation]

    res = _npe2.read(paths, plugin, stack=stack)
    if res is not None:
        _ld, hookimpl = res
        return [] if _is_null_layer_sentinel(_ld) else _ld, hookimpl  # type: ignore [return-value]

    hook_caller = plugin_manager.hook.napari_get_reader
    paths = [abspath_or_url(p, must_exist=True) for p in paths]
    if not plugin and not stack:
        extension = os.path.splitext(paths[0])[-1]
        plugin = plugin_manager.get_reader_for_extension(extension)

    # npe1 compact whether we are reading as stack or not is carried in the type
    # of paths
    npe1_path = paths if stack else paths[0]
    hookimpl = None
    if plugin:
        if plugin == 'napari':
            # napari is npe2 only
            message = trans._(
                'No plugin found capable of reading {repr_path!r}.',
                deferred=True,
                repr_path=npe1_path,
            )
            raise ValueError(message)

        if plugin not in plugin_manager.plugins:
            names = {i.plugin_name for i in hook_caller.get_hookimpls()}
            raise ValueError(
                trans._(
                    "There is no registered plugin named '{plugin}'.\nNames of plugins offering readers are: {names}",
                    deferred=True,
                    plugin=plugin,
                    names=names,
                )
            )
        reader = hook_caller._call_plugin(plugin, path=npe1_path)
        if not callable(reader):
            raise ValueError(
                trans._(
                    'Plugin {plugin!r} does not support file(s) {paths}',
                    deferred=True,
                    plugin=plugin,
                    paths=paths,
                )
            )

        hookimpl = hook_caller.get_plugin_implementation(plugin)
        layer_data = reader(npe1_path)
        # if the reader returns a "null layer" sentinel indicating an empty
        # file, return an empty list, otherwise return the result or None
        if _is_null_layer_sentinel(layer_data):
            return [], hookimpl

        return layer_data or None, hookimpl

    layer_data = None
    result = hook_caller.call_with_result_obj(path=npe1_path)
    if reader := result.result:  # will raise exceptions if any occurred
        try:
            layer_data = reader(npe1_path)  # try to read data
            hookimpl = result.implementation
        except Exception as exc:
            raise PluginCallError(result.implementation, cause=exc) from exc

    if not layer_data:
        # if layer_data is empty, it means no plugin could read path
        # we just want to provide some useful feedback, which includes
        # whether or not paths were passed to plugins as a list.
        if stack:
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

        # TODO: change to a warning notification in a later PR
        raise ValueError(message)

    # if the reader returns a "null layer" sentinel indicating an empty file,
    # return an empty list, otherwise return the result or None
    _data = [] if _is_null_layer_sentinel(layer_data) else layer_data or None
    return _data, hookimpl


def save_layers(
    path: str,
    layers: List[Layer],
    *,
    plugin: Optional[str] = None,
    _writer: Optional[WriterContribution] = None,
) -> List[str]:
    """Write list of layers or individual layer to a path using writer plugins.

    If ``plugin`` is not provided and only one layer is passed, then we
    directly call ``plugin_manager.hook.napari_write_<layer>()`` which
    will loop through implementations and stop when the first one returns a
    non-None result. The order in which implementations are called can be
    changed with the hook ``bring_to_front`` method, for instance:
    ``plugin_manager.hook.napari_write_points.bring_to_front``

    If ``plugin`` is not provided and multiple layers are passed, then
    we call ``plugin_manager.hook.napari_get_writer()`` which loops through
    plugins to find the first one that knows how to handle the combination of
    layers and is able to write the file. If no plugins offer
    ``napari_get_writer`` for that combination of layers then the builtin
    ``napari_get_writer`` implementation will create a folder and call
    ``napari_write_<layer>`` for each layer using the ``layer.name`` variable
    to modify the path such that the layers are written to unique files in the
    folder.

    If ``plugin`` is provided and a single layer is passed, then
    we call the ``napari_write_<layer_type>`` for that plugin, and if it
    fails we error.

    If a ``plugin`` is provided and multiple layers are passed, then
    we call we call ``napari_get_writer`` for that plugin, and if it
    doesn’t return a WriterFunction we error, otherwise we call it and if
    that fails if it we error.

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
    plugin. If it doesn’t return a ``WriterFunction`` we error, otherwise we
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

    # Try to use NPE2 first
    written_paths = _npe2.write_layers(path, layers, plugin_name, _writer)
    if written_paths:
        return written_paths
    logger.debug("Falling back to original plugin engine.")

    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    layer_types = [ld[2] for ld in layer_data]

    if not plugin_name and isinstance(path, (str, pathlib.Path)):
        extension = os.path.splitext(path)[-1]
        plugin_name = plugin_manager.get_writer_for_extension(extension)

    hook_caller = plugin_manager.hook.napari_get_writer
    path = abspath_or_url(path)
    logger.debug(f"Writing to {path}.  Hook caller: {hook_caller}")
    if plugin_name:
        # if plugin has been specified we just directly call napari_get_writer
        # with that plugin_name.
        if plugin_name not in plugin_manager.plugins:
            names = {i.plugin_name for i in hook_caller.get_hookimpls()}
            raise ValueError(
                trans._(
                    "There is no registered plugin named '{plugin_name}'.\nNames of plugins offering writers are: {names}",
                    deferred=True,
                    plugin_name=plugin_name,
                    names=names,
                )
            )
        implementation = hook_caller.get_plugin_implementation(plugin_name)
        writer_function = hook_caller(
            _plugin=plugin_name, path=path, layer_types=layer_types
        )
    else:
        result = hook_caller.call_with_result_obj(
            path=path, layer_types=layer_types, _return_impl=True
        )
        writer_function = result.result
        implementation = result.implementation

    if not callable(writer_function):
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

    try:
        return writer_function(abspath_or_url(path), layer_data)
    except Exception as exc:
        raise PluginCallError(implementation, cause=exc)


def _write_single_layer_with_plugins(
    path: str,
    layer: Layer,
    *,
    plugin_name: Optional[str] = None,
    _writer: Optional[WriterContribution] = None,
) -> Optional[str]:
    """Write single layer data with a plugin.

    If ``plugin_name`` is not provided then we just directly call
    ``plugin_manager.hook.napari_write_<layer>()`` which will loop through
    implementations and stop when the first one returns a non-None result. The
    order in which implementations are called can be changed with the
    implementation sorter/disabler.

    If ``plugin_name`` is provided, then we call the
    ``napari_write_<layer_type>`` for that plugin, and if it fails we error.

    Exceptions will be caught and stored as PluginErrors
    (in plugins.exceptions.PLUGIN_ERRORS)

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

    # Try to use NPE2 first
    written_paths = _npe2.write_layers(path, [layer], plugin_name, _writer)
    if written_paths:
        return written_paths[0]
    logger.debug("Falling back to original plugin engine.")

    hook_caller = getattr(
        plugin_manager.hook, f'napari_write_{layer._type_string}'
    )

    if not plugin_name and isinstance(path, (str, pathlib.Path)):
        extension = os.path.splitext(path)[-1]
        plugin_name = plugin_manager.get_writer_for_extension(extension)

    logger.debug(f"Writing to {path}.  Hook caller: {hook_caller}")
    if plugin_name and (plugin_name not in plugin_manager.plugins):
        names = {i.plugin_name for i in hook_caller.get_hookimpls()}
        raise ValueError(
            trans._(
                "There is no registered plugin named '{plugin_name}'.\nPlugins capable of writing layer._type_string layers are: {names}",
                deferred=True,
                plugin_name=plugin_name,
                names=names,
            )
        )

    # Call the hook_caller
    return hook_caller(
        _plugin=plugin_name,
        path=abspath_or_url(path),
        data=layer.data,
        meta=layer._get_state(),
    )
