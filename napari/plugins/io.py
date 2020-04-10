from . import PluginError, plugin_manager as napari_plugin_manager
from ._hook_callers import execute_hook
from typing import Optional, Union, Sequence
from ..types import LayerData
from logging import getLogger

logger = getLogger(__name__)


def read_data_with_plugins(
    path: Union[str, Sequence[str]], plugin_manager=None
) -> Optional[LayerData]:
    """Iterate reader hooks and return first non-None LayerData or None.

    This function returns as soon as the path has been read successfully,
    while catching any plugin exceptions, storing them for later retrievial,
    providing useful error messages, and relooping until either layer data is
    returned, or no valid readers are found.

    Exceptions will be caught and stored as PluginErrors
    (in plugins.PLUGIN_ERRORS)

    Parameters
    ----------
    path : str
        The path (file, directory, url) to open.
    plugin_manager : pluggy.PluginManager, optional
        Instance of a pluggy PluginManager.  by default the main napari
        plugin_manager will be used.

    Returns
    -------
    LayerData or None
        LayerData that can be *passed to _add_layer_from_data.  If no reader
        plugins are (or they all error), returns None.
    """
    plugin_manager = plugin_manager or napari_plugin_manager
    skip_impls = []
    while True:
        (reader, implementation) = execute_hook(
            plugin_manager.hook.napari_get_reader,
            path=path,
            return_impl=True,
            skip_impls=skip_impls,
        )
        if not reader:
            # we're all out of reader plugins
            return None
        try:
            return reader(path)  # try to read the data.
        except Exception as exc:
            # If execute_hook did return a reader, but the reader then failed
            # while trying to read the path, we store the traceback for later
            # retrieval, warn the user, and continue looking for readers
            # (skipping this one)
            msg = (
                f"Error in plugin '{implementation.plugin_name}', "
                "hook 'napari_get_reader'"
            )
            err = PluginError(
                msg, implementation.plugin_name, implementation.plugin.__name__
            )
            err.__cause__ = exc  # like `raise PluginError() from exc`
            # store the exception for later retrieval
            plugin_manager._exceptions[implementation.plugin_name].append(err)
            skip_impls.append(implementation)  # don't try this impl again
            if implementation.plugin_name != 'builtins':
                # If builtins doesn't work, they will get a "no reader" found
                # error anyway, so it looks a bit weird to show them that the
                # "builtin plugin" didn't work.
                logger.error(err.format_with_contact_info())


def write_data_with_plugins(
    path: str, layer_data: LayerData, plugin_manager=None,
):
    """Iterate writer hooks and write data with the first successful writer.

    This function returns as soon as the data has been written successfully,
    while catching any plugin exceptions, storing them for later retrievial,
    providing useful error messages, and relooping until either data is
    writen, or no valid writers are found.

    Exceptions will be caught and stored as PluginErrors
    (in plugin_manager._exceptions)

    Parameters
    ----------
    path : str
        The path (file, directory, url) to save.
    layer_types : list of str
        List of layer types that will be provided to the writer function.
    plugin_manager : pluggy.PluginManager, optional
        Instance of a pluggy PluginManager.  by default the main napari
        plugin_manager will be used.
    """
    layer_types = [single_layer_data[2] for single_layer_data in layer_data]
    plugin_manager = plugin_manager or napari_plugin_manager
    skip_impls = []
    while True:
        (writer, implementation) = execute_hook(
            plugin_manager.hook.napari_get_writer,
            path=path,
            layer_types=layer_types,
            return_impl=True,
            skip_impls=skip_impls,
        )
        if not writer:
            # we're all out of writer plugins
            return None
        try:
            return writer(path, layer_data)  # try to write the data.
        except Exception as exc:
            # If execute_hook did return a writer, but the writer then failed
            # while trying to write the path, we store the traceback for later
            # retrieval, warn the user, and continue looking for writers
            # (skipping this one)
            msg = (
                f"Error in plugin '{implementation.plugin_name}', "
                "hook 'napari_get_writer'"
            )
            # instantiating this PluginError stores it in
            # plugins.exceptions.PLUGIN_ERRORS, where it can be retrieved later
            err = PluginError(
                msg, implementation.plugin_name, implementation.plugin.__name__
            )
            err.__cause__ = exc  # like `raise PluginError() from exc`

            skip_impls.append(implementation)  # don't try this impl again
            if implementation.plugin_name != 'builtins':
                # If builtins doesn't work, they will get a "no writer" found
                # error anyway, so it looks a bit weird to show them that the
                # "builtin plugin" didn't work.
                logger.error(err.format_with_contact_info())
