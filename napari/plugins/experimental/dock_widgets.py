from typing import List, Tuple

from qtpy.QtWidgets import QWidget

from .. import plugin_manager as napari_plugin_manager


def get_dock_widgets_from_plugin(
    plugin, viewer, plugin_manager=napari_plugin_manager,
) -> List[Tuple[QWidget, dict]]:
    """Get a list of dock widgets from a plugin.

    Parameters
    ----------
    plugin : str
        Name of the plugin to get dock widgets from.
    viewer : napari.Viewer
        napari Viewer object.
    plugin_manager : plugins.PluginManager, optional
        Instance of a napari PluginManager.  by default the main napari
        plugin_manager will be used.

    Returns
    -------
    dock_widgets : list
        List of 2-tuples, where each tuple has a widget and dictionary of
        keyword arguments for the viewer.window.add_dock_widget method.
    """
    plugin_name = plugin

    hook_caller = plugin_manager.hook.napari_experimental_provide_dock_widget

    if plugin_name not in plugin_manager.plugins:
        names = {i.plugin_name for i in hook_caller.get_hookimpls()}
        raise ValueError(
            f"There is no registered plugin named '{plugin_name}'.\n"
            "Plugins capable of providing dock widgets "
            f"are: {names}"
        )

    # Call the hook_caller
    return hook_caller(_plugin=plugin_name, viewer=viewer)
