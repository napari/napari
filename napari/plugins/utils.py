from . import PluginError, log_plugin_error
from . import plugin_manager as napari_plugin_manager
from ._hookexec import _hookexec
from typing import Optional
from ..types import LayerData


def get_layer_data_from_plugins(
    path: str, plugin_manager=None
) -> Optional[LayerData]:
    """Iterate reader hooks and return first non-None LayerData or None.

    This function returns as soon as the path has been read successfully,
    while catching any plugin exceptions, storing them for later retrievial,
    providing useful error messages, and relooping until either layer data is
    returned, or no readers are found.

    Exceptions will be caught and stored as PluginErrors
    (in plugin_manager._exceptions)

    Parameters
    ----------
    path : str
        The path (file, directory, url) to open
    plugin_manager : pluggy.PluginManager, optional
        Instance of a pluggy PluginManager.  by default the main napari
        plugin_manager will be used.

    Returns
    -------
    LayerData or None
        LayerData that can be *passed to _add_layer_from_data.  If no reader
        plugins are (or they all error), returns None
    """
    plugin_manager = plugin_manager or napari_plugin_manager
    skip_imps = []
    while True:
        (reader, imp) = _hookexec(
            plugin_manager.hook.napari_get_reader,
            path=path,
            with_impl=True,
            skip_imps=skip_imps,
        )
        if not reader:
            # we're all out of reader plugins
            return None
        try:
            return reader(path)  # try to read the data.
        except Exception as exc:
            # If _hookexec did return a reader, but the reader then failed
            # while trying to read the path, we store the traceback for later
            # retrieval, warn the user, and continue looking for readers
            # (skipping this one)
            msg = (
                f"Error in plugin '{imp.plugin_name}', "
                "hook 'napari_get_reader'"
            )
            err = PluginError(msg, imp.plugin_name, imp.plugin.__name__)
            err.__cause__ = exc  # like `raise PluginError() from exc`
            # store the exception for later retrieval
            plugin_manager._exceptions[imp.plugin_name].append(err)
            skip_imps.append(imp)  # don't try this impl again
            if imp.plugin_name != 'builtins':
                # If builtins doesn't work, they will get a "no reader" found
                # error anyway, so it looks a bit weird to show them that the
                # "builtin plugin" didn't work.
                log_plugin_error(err)  # let the user know
