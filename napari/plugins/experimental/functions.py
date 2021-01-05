from typing import Callable, Dict, Tuple

from .. import plugin_manager as napari_plugin_manager


def get_functions_from_plugin(
    plugin,
    plugin_manager=napari_plugin_manager,
) -> Tuple[Tuple[Callable, Dict, Dict]]:
    """Get a list of functions from a plugin.

    Parameters
    ----------
    plugin : str
        Name of the plugin to get dock widgets from.
    plugin_manager : plugins.PluginManager, optional
        Instance of a napari PluginManager.  by default the main napari
        plugin_manager will be used.

    Returns
    -------
    functions : tuple
        Tuple of 3-tuple, where each tuple has a function a dictionary of
        keyword arguments for magicgui, and a dictionary of keyword arguments
        for the viewer.window.add_dock_widget method.
    """
    plugin_name = plugin

    hook_caller = plugin_manager.hook.napari_experimental_provide_functions

    if plugin_name not in plugin_manager.plugins:
        names = {i.plugin_name for i in hook_caller.get_hookimpls()}
        raise ValueError(
            f"There is no registered plugin named '{plugin_name}'.\n"
            "Plugins capable of providing dock widgets "
            f"are: {names}"
        )

    # Call the hook_caller
    return hook_caller(_plugin=plugin_name)
