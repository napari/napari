from logging import getLogger
from typing import Optional, Sequence, Union, List

from pluggy.hooks import HookImpl

from ..types import LayerData
from . import PluginError
from . import plugin_manager as napari_plugin_manager

logger = getLogger(__name__)


def read_data_with_plugins(
    path: Union[str, Sequence[str]],
    plugin: Optional[str] = None,
    plugin_manager=napari_plugin_manager,
) -> Optional[LayerData]:
    """Iterate reader hooks and return first non-None LayerData or None.

    This function returns as soon as the path has been read successfully,
    while catching any plugin exceptions, storing them for later retrievial,
    providing useful error messages, and relooping until either layer data is
    returned, or no readers are found.

    Exceptions will be caught and stored as PluginErrors
    (in plugins.exceptions.PLUGIN_ERRORS)

    Parameters
    ----------
    path : str
        The path (file, directory, url) to open
    plugin : str, optional
        Name of a plugin to use.  If provided, will force ``path`` to be read
        with the specified ``plugin``.  If the requested plugin cannot read
        ``path``, a PluginCallError will be raised.
    plugin_manager : plugins.PluginManager, optional
        Instance of a napari PluginManager.  by default the main napari
        plugin_manager will be used.

    Returns
    -------
    LayerData : list of tuples, or None
        LayerData that can be passed to :func:`Viewer._add_layer_from_data()
        <napari.components.add_layers_mixin.AddLayersMixin._add_layer_from_data>`.
        ``LayerData`` is a list tuples, where each tuple is one of
        ``(data,)``, ``(data, meta)``, or ``(data, meta, layer_type)`` .

        If no reader plugins are (or they all error), returns ``None``

    Raises
    ------
    PluginCallError
        If ``plugin`` is specified but raises an Exception while reading.
    """
    hook_caller = plugin_manager.hook.napari_get_reader

    if plugin:
        reader = hook_caller._call_plugin(plugin, path=path)
        return reader(path)

    skip_impls: List[HookImpl] = []
    while True:
        result = hook_caller.call_with_result_obj(
            path=path, _skip_impls=skip_impls
        )
        reader = result.result  # will raise exceptions if any occured
        if not reader:
            # we're all out of reader plugins
            return None
        try:
            return reader(path)  # try to read data
        except Exception as exc:
            # If the hook did return a reader, but the reader then failed
            # while trying to read the path, we store the traceback for later
            # retrieval, warn the user, and continue looking for readers
            # (skipping this one)
            hook_implementation = result.implementation
            plugin_name = hook_implementation.plugin_name
            plugin_module = hook_implementation.plugin.__name__
            msg = (
                f"Error in plugin '{plugin_name}', "
                f"hook 'napari_get_reader': {exc}"
            )
            # instantiating this PluginError stores it in
            # plugins.exceptions.PLUGIN_ERRORS, where it can be retrieved later
            err = PluginError(msg, plugin_name, plugin_module)
            err.__cause__ = exc  # like ``raise PluginError() from exc``

            skip_impls.append(hook_implementation)  # don't try this impl again
            if plugin_name != 'builtins':
                # If builtins doesn't work, they will get a "no reader" found
                # error anyway, so it looks a bit weird to show them that the
                # "builtin plugin" didn't work.
                logger.error(err.format_with_contact_info())
