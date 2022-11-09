from napari_plugin_engine import PluginError, standard_metadata

from napari.utils.translations import trans


def format_exceptions(
    plugin_name: str, as_html: bool = False, color="Neutral"
):
    """Return formatted tracebacks for all exceptions raised by plugin.

    Parameters
    ----------
    plugin_name : str
        The name of a plugin for which to retrieve tracebacks.
    as_html : bool
        Whether to return the exception string as formatted html,
        defaults to False.

    Returns
    -------
    str
        A formatted string with traceback information for every exception
        raised by ``plugin_name`` during this session.
    """
    _plugin_errors = PluginError.get(plugin_name=plugin_name)
    if not _plugin_errors:
        return ''

    from napari import __version__
    from napari.utils._tracebacks import get_tb_formatter

    format_exc_info = get_tb_formatter()

    _linewidth = 80
    _pad = (_linewidth - len(plugin_name) - 18) // 2
    msg = [
        trans._(
            "{pad} Errors for plugin '{plugin_name}' {pad}",
            deferred=True,
            pad='=' * _pad,
            plugin_name=plugin_name,
        ),
        '',
        f'{"napari version": >16}: {__version__}',
    ]

    err0 = _plugin_errors[0]
    if err0.plugin:
        package_meta = standard_metadata(err0.plugin)
        if package_meta:
            msg.extend(
                [
                    f'{"plugin package": >16}: {package_meta["package"]}',
                    f'{"version": >16}: {package_meta["version"]}',
                    f'{"module": >16}: {err0.plugin}',
                ]
            )
    msg.append('')

    for n, err in enumerate(_plugin_errors):
        _pad = _linewidth - len(str(err)) - 10
        msg += ['', f'ERROR #{n + 1}:  {str(err)} {"-" * _pad}', '']
        msg.append(format_exc_info(err.info(), as_html, color))

    msg.append('=' * _linewidth)

    return ("<br>" if as_html else "\n").join(msg)
