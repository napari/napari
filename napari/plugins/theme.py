"""Handling of theme data by the hook specification."""
from typing import Dict, List, Optional, Tuple
from logging import getLogger
from pathlib import Path

from napari_plugin_engine import HookImplementation, PluginCallError
from . import plugin_manager
from ..utils.translations import trans

logger = getLogger(__name__)


def get_stylesheet_from_plugins(
    plugin: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
    """Iterate through hooks and return stylesheet."""

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


def register_plugin_themes(plugin: Optional[str] = None):
    """Register plugin theme data."""
    from .._qt.qt_resources import _register_napari_resources

    # get data from each plugin (or single specified plugin)
    qss_files, svg_paths, theme_colors = get_stylesheet_from_plugins(plugin)

    force_rebuild = False
    # register new themes
    if theme_colors:
        from ..utils.theme import register_theme

        for name, theme in theme_colors.items():
            register_theme(name, theme)
        force_rebuild = True

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
