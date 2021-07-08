"""Handling of theme data by the hook specification."""
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from napari_plugin_engine import HookImplementation, PluginCallError

from ..utils.translations import trans
from . import plugin_manager

logger = getLogger(__name__)


def get_qss_from_plugins(plugin: Optional[str] = None) -> List[str]:
    """Iterate through hooks and return stylesheet(s))."""
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
            theme_caller = hook_caller._call_plugin(plugin)
            if not callable(theme_caller):
                raise ValueError(
                    trans._(
                        'Plugin {plugin!r} is not callable.',
                        deferred=True,
                        plugin=plugin,
                    )
                )
            qss_files = theme_caller()
            return qss_files

    errors: List[PluginCallError] = []
    skip_impls: List[HookImplementation] = []
    qss_files = []
    while True:
        result = hook_caller.call_with_result_obj(_skip_impls=skip_impls)
        theme_caller = result.result  # will raise exceptions if any occurred
        if not theme_caller:
            # we're all out of theme plugins
            break
        try:
            _qss_files = theme_caller()
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


def get_icons_from_plugins(plugin: Optional[str] = None) -> List[str]:
    """Iterate through hooks and return stylesheet(s))."""
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
            theme_caller = hook_caller._call_plugin(plugin)
            if not callable(theme_caller):
                raise ValueError(
                    trans._(
                        'Plugin {plugin!r} is not callable.',
                        deferred=True,
                        plugin=plugin,
                    )
                )
            svg_paths = theme_caller()
            return svg_paths

    errors: List[PluginCallError] = []
    skip_impls: List[HookImplementation] = []
    svg_paths = []
    while True:
        result = hook_caller.call_with_result_obj(_skip_impls=skip_impls)
        theme_caller = result.result  # will raise exceptions if any occurred
        if not theme_caller:
            # we're all out of theme plugins
            break
        try:
            _svg_paths = theme_caller()
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


def get_stylesheet_from_plugins(
    plugin: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
    """Iterate through hooks and return stylesheet.

    This function returns a list of qss files, list of svg icons and
    dictionary of themes. If multiple plugins implement the theme
    hook, all of their data will be loaded.

    Parameters
    ----------
    plugin : str, optional
        Name of a plugin to use. If provided, will force the theme data to be
        provided by the specified ``plugin``. If the requested plugin cannot
        provide the data, a PluginCallError will be raised.

    Returns
    -------
    qss_files : List[str]
        A list of Qt stylesheets with the file extension .qss. Napari provides
        several default stylesheets with names `00_base.qss`, `01_buttons.qss` etc
        which are first sorted (hence the number at the front) and then progressively
        read and appended to single stylesheet. You can provide your own stylesheets
        that override the napari defaults by creating a new stylesheet with progressively
        larger name.
    svg_paths : List[str]
        A list of svg files to be colorized and used in napari. These can be new icons that
        are required by your own plugin or icons to replace the currently available icons.
    color_dict : Dict[str, Dict[str, str]
        A dictionary containing new color scheme to be used by napari. You can replace
        existing themes by using the same names.

    Raises
    ------
    PluginCallError
        If ``plugin`` is specified but raises an Exception while executing.
    """

    hook_caller = plugin_manager.hook.napari_experimental_provide_theme
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
            theme_caller = hook_caller._call_plugin(plugin)
            if not callable(theme_caller):
                raise ValueError(
                    trans._(
                        'Plugin {plugin!r} is not callable.',
                        deferred=True,
                        plugin=plugin,
                    )
                )
            qss_files, svg_paths, theme_colors = theme_caller()
            return qss_files, svg_paths, theme_colors

    errors: List[PluginCallError] = []
    skip_impls: List[HookImplementation] = []
    qss_files, svg_paths, theme_colors = [], [], {}
    while True:
        result = hook_caller.call_with_result_obj(_skip_impls=skip_impls)
        theme_caller = result.result  # will raise exceptions if any occurred
        if not theme_caller:
            # we're all out of theme plugins
            break
        try:
            _qss_files, _svg_paths, _theme_colors = theme_caller()
            if _qss_files:
                qss_files.extend(_qss_files)
            if _svg_paths:
                svg_paths.extend(_svg_paths)
            if _theme_colors:
                theme_colors.update(**_theme_colors)
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
    return qss_files, svg_paths, theme_colors


def register_plugin_resources(plugin: Optional[str] = None):
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

        for icon in svg_paths:
            icon = Path(icon)
            if icon.suffix == ".svg":
                ICONS[icon.stem] = str(icon)
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
