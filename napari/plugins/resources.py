"""Handling of theme data by the hook specification."""
import typing as ty
from logging import getLogger
from pathlib import Path

from napari_plugin_engine import HookImplementation, PluginCallError

from ..utils.translations import trans
from . import plugin_manager

logger = getLogger(__name__)


def get_qss_from_plugins(plugin: ty.Optional[str] = None) -> ty.List[str]:
    """Iterate through hooks and return stylesheet(s).

    This function returns list of paths to qss files. These can then
    be used to build the stylesheet used by napari for instance to
    override existing style or add additional styles to customize
    your own plugin or application.

    Parameters
    ----------
    plugin : str, optional
        Name of the plugin to get stylesheets from. If one is not provided
        then all plugins that use this hook will provide qss stylesheets.

    Returns
    -------
    list of str
        File paths of any qss stylesheets that should be used by napari.
    """
    hook_caller = plugin_manager.hook.napari_experimental_provide_qss
    if plugin:
        if plugin not in plugin_manager.plugins:
            if plugin not in plugin_manager.plugins:
                names = {i.plugin_name for i in hook_caller.get_hookimpls()}
                raise ValueError(
                    trans._(
                        "There is no registered plugin named '{plugin}'."
                        "\nNames of plugins offering readers are: {names}",
                        deferred=True,
                        plugin=plugin,
                        names=names,
                    )
                )
            qss_caller = hook_caller._call_plugin(plugin)
            if not callable(qss_caller):
                raise ValueError(
                    trans._(
                        'Plugin {plugin!r} is not callable.',
                        deferred=True,
                        plugin=plugin,
                    )
                )
            qss_files = qss_caller()
            return qss_files

    errors: ty.List[PluginCallError] = []
    skip_impls: ty.List[HookImplementation] = []
    qss_files = []
    while True:
        result = hook_caller.call_with_result_obj(_skip_impls=skip_impls)
        qss_caller = result.result  # will raise exceptions if any occurred
        if not qss_caller:
            # we're all out of theme plugins
            break
        try:
            _qss_files = qss_caller()
            if _qss_files:
                qss_files.extend(_qss_files)
        except Exception as exc:
            # collect the error and log it, but don't raise it.
            err = PluginCallError(result.implementation, cause=exc)
            err.log(logger=logger)
            errors.append(err)
        # don't try this impl again
        skip_impls.append(result.implementation)

    if errors:
        names = {repr(e.plugin_name) for e in errors}
        err_msg = f"({len(errors)}) error{'s' if len(errors) > 1 else ''} "
        err_msg += f"occurred in plugins: {', '.join(names)}. "
        err_msg += 'See full error logs in "Plugins → Plugin Errors..."'
        logger.error(err_msg)
    return qss_files


def get_icons_from_plugins(
    plugin: ty.Optional[str] = None,
) -> ty.List[ty.Tuple[str, str]]:
    """Iterate through hooks and return paths to svg icons.

    This function returns list of paths to svg icon files. These can
    be used to override existing icons or provide additional icons
    that can be used by your own plugin or application.

    Parameters
    ----------
    plugin : str, optional
        Name of the plugin to get icons from. If one is not provided
        then all plugins that use this hook will provide icons.

    Returns
    -------
    list of str
        File paths of any svg icons that should be used by napari.
    """
    hook_caller = plugin_manager.hook.napari_experimental_provide_icons
    if plugin:
        if plugin not in plugin_manager.plugins:
            if plugin not in plugin_manager.plugins:
                names = {i.plugin_name for i in hook_caller.get_hookimpls()}
                raise ValueError(
                    trans._(
                        "There is no registered plugin named '{plugin}'."
                        "\nNames of plugins offering readers are: {names}",
                        deferred=True,
                        plugin=plugin,
                        names=names,
                    )
                )
            icon_caller = hook_caller._call_plugin(plugin)
            if not callable(icon_caller):
                raise ValueError(
                    trans._(
                        'Plugin {plugin!r} is not callable.',
                        deferred=True,
                        plugin=plugin,
                    )
                )
            svg_paths = icon_caller()
            return svg_paths

    errors: ty.List[PluginCallError] = []
    skip_impls: ty.List[HookImplementation] = []
    svg_paths = []
    while True:
        result = hook_caller.call_with_result_obj(_skip_impls=skip_impls)
        icon_caller = result.result  # will raise exceptions if any occurred
        if not icon_caller:
            # we're all out of theme plugins
            break
        try:
            plugin_name = result.implementation.plugin_name
            _svg_paths = icon_caller()
            _svg_paths = [(plugin_name, icon) for icon in _svg_paths]
            if _svg_paths:
                svg_paths.extend(_svg_paths)
        except Exception as exc:
            # collect the error and log it, but don't raise it.
            err = PluginCallError(result.implementation, cause=exc)
            err.log(logger=logger)
            errors.append(err)
        # don't try this impl again
        skip_impls.append(result.implementation)

    if errors:
        names = {repr(e.plugin_name) for e in errors}
        err_msg = f"({len(errors)}) error{'s' if len(errors) > 1 else ''} "
        err_msg += f"occurred in plugins: {', '.join(names)}. "
        err_msg += 'See full error logs in "Plugins → Plugin Errors..."'
        logger.error(err_msg)
    return svg_paths


def register_plugin_resources(plugin: ty.Optional[str] = None):
    """Register plugin theme data.

    This function will load the theme data and update dictionary
    of icons, stylesheets and themes. If any of the plugins have
    returned the data, napari resources will be forcefully refreshed.
    """
    from .._qt.qt_resources import _register_napari_resources

    # get data from each plugin (or single specified plugin)
    qss_files = get_qss_from_plugins(plugin)
    svg_paths = get_icons_from_plugins(plugin)

    force_rebuild = False

    # register icons
    if svg_paths:
        from ..resources._icons import ICONS

        for plugin_name, icon in svg_paths:
            icon = Path(icon)
            if icon.suffix == ".svg":
                ICONS[f"{plugin_name}/{icon.stem}"] = str(icon)
        force_rebuild = True
    # register qss files
    if qss_files:
        from .._qt.qt_resources import STYLES

        for style in qss_files:
            style = Path(style)
            if style.suffix == ".qss":
                STYLES[style.stem] = str(style)
        force_rebuild = True

    # since there were some changes to the underlying icons/stylesheets/theme coloring
    # the resources should be regenerated
    _register_napari_resources(force_rebuild=force_rebuild)
