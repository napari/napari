import sys
from collections import defaultdict
from types import TracebackType
from typing import DefaultDict, Dict, List, Optional, Tuple, Type
import re
import IPython.core.ultratb


# This is a mapping of plugin_name -> PluginError instances
# all PluginErrors get added to this in PluginError.__init__
PLUGIN_ERRORS: DefaultDict[str, List['PluginError']] = defaultdict(list)


if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


class PluginError(Exception):
    """Base class for all plugin-related errors.

    Instantiating a PluginError (whether raised or not), adds the exception
    instance to the PLUGIN_ERRORS dict for later retrieval.

    Parameters
    ----------
    message : str
        A message for the exception
    plugin_name : str
        The name of the plugin that had the error
    plugin_module : str
        The module of the plugin that had the error
    """

    def __init__(
        self, message: str, plugin_name: str, plugin_module: str
    ) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name
        self.plugin_module = plugin_module
        PLUGIN_ERRORS[plugin_name].append(self)

    def format_with_contact_info(self) -> str:
        """Make formatted string with context and contact info if possible."""
        # circular imports
        from napari import __version__

        msg = f'\n\nPluginError: {self}'
        msg += '\n(Use "Plugins > Plugin errors..." to review/report errors.)'
        if self.__cause__:
            cause = str(self.__cause__).replace("\n", "\n" + " " * 13)
            msg += f'\n  Cause was: {cause}'
        contact = fetch_module_metadata(self.plugin_module)
        if contact:
            extra = [f'{k: >11}: {v}' for k, v in contact.items()]
            extra += [f'{"napari": >11}: v{__version__}']
            msg += "\n".join(extra)
        msg += '\n'
        return msg

    def info(
        self,
    ) -> Tuple[Type['PluginError'], 'PluginError', Optional[TracebackType]]:
        """Return info as would be returned from sys.exc_info()."""
        return (self.__class__, self, self.__traceback__)


class PluginImportError(PluginError, ImportError):
    """Raised when a plugin fails to import."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f"Failed to import plugin: '{plugin_name}'"
        super().__init__(msg, plugin_name, plugin_module)


class PluginRegistrationError(PluginError):
    """Raised when a plugin fails to register with pluggy."""

    def __init__(self, plugin_name: str, plugin_module: str) -> None:
        msg = f"Failed to register plugin: '{plugin_name}'"
        super().__init__(msg, plugin_name, plugin_module)


def format_exceptions(plugin_name: str, as_html: bool = False) -> str:
    """Return formatted tracebacks for all exceptions raised by plugin.

    This method takes advantage of ``IPython.core.ultratb.VerboseTB`` to create
    a nicely formatted traceback.  If ``as_html`` is True, is then converted
    from ansi to html.  (This approach was used instead of cgitb because the
    IPython ultratb tracebacks were preferred).

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
    from napari import __version__

    if not PLUGIN_ERRORS.get(plugin_name):
        return ''

    color = 'Linux' if as_html else 'NoColor'
    vbtb = IPython.core.ultratb.VerboseTB(color_scheme=color)

    _linewidth = 80
    _pad = (_linewidth - len(plugin_name) - 18) // 2
    msg = [
        f"{'=' * _pad} Errors for plugin '{plugin_name}' {'=' * _pad}",
        '',
        f'{"napari version": >16}: {__version__}',
    ]
    try:
        err0 = PLUGIN_ERRORS[plugin_name][0]
        package_meta = fetch_module_metadata(err0.plugin_module)
        if package_meta:
            msg.extend(
                [
                    f'{"plugin name": >16}: {package_meta["name"]}',
                    f'{"version": >16}: {package_meta["version"]}',
                    f'{"module": >16}: {err0.plugin_module}',
                ]
            )
    except Exception:
        pass
    msg.append('')

    for n, err in enumerate(PLUGIN_ERRORS.get(plugin_name, [])):
        _pad = _linewidth - len(str(err)) - 10
        msg += ['', f'ERROR #{n + 1}:  {str(err)} {"-" * _pad}', '']
        etype, value, tb = err.info()
        if as_html:
            ansi_string = vbtb.text(etype, value, tb)
            html = "".join(ansi2html(ansi_string.replace(" ", "&nbsp;")))
            html = html.replace("\n", "<br>")
            html = (
                "<span style='font-family: monaco,courier,monospace;'>"
                + html
                + "</span>"
            )
            msg.append(html)
        else:
            msg.append(vbtb.text(etype, value, tb))

    msg.append('=' * _linewidth)

    return ("<br>" if as_html else "\n").join(msg)


def fetch_module_metadata(distname: str) -> Optional[Dict[str, str]]:
    """Attempt to retrieve name, version, contact email & url for a package.

    Parameters
    ----------
    distname : str
        Name of a distribution.  Note: this must match the *name* of the
        package in the METADATA file... not the name of the module.

    Returns
    -------
    package_info : dict or None
        A dict with keys 'name', 'version', 'email', and 'url'.
        Returns None of the distname cannot be found.
    """
    try:
        meta = importlib_metadata.metadata(distname)
    except importlib_metadata.PackageNotFoundError:
        return None
    return {
        'name': meta.get('Name'),
        'version': meta.get('Version'),
        'email': meta.get('Author-Email') or meta.get('Maintainer-Email'),
        'url': meta.get('Home-page') or meta.get('Download-Url'),
    }


ANSI_STYLES = {
    1: {"font_weight": "bold"},
    2: {"font_weight": "lighter"},
    3: {"font_weight": "italic"},
    4: {"text_decoration": "underline"},
    5: {"text_decoration": "blink"},
    6: {"text_decoration": "blink"},
    8: {"visibility": "hidden"},
    9: {"text_decoration": "line-through"},
    30: {"color": "black"},
    31: {"color": "red"},
    32: {"color": "green"},
    33: {"color": "yellow"},
    34: {"color": "blue"},
    35: {"color": "magenta"},
    36: {"color": "cyan"},
    37: {"color": "white"},
}


def ansi2html(ansi_string, styles=ANSI_STYLES):
    previous_end = 0
    in_span = False
    ansi_codes = []
    ansi_finder = re.compile("\033\\[" "([\\d;]*)" "([a-zA-z])")
    for match in ansi_finder.finditer(ansi_string):
        yield ansi_string[previous_end : match.start()]
        previous_end = match.end()
        params, command = match.groups()

        if command not in "mM":
            continue

        try:
            params = [int(p) for p in params.split(";")]
        except ValueError:
            params = [0]

        for i, v in enumerate(params):
            if v == 0:
                params = params[i + 1 :]
                if in_span:
                    in_span = False
                    yield "</span>"
                ansi_codes = []
                if not params:
                    continue

        ansi_codes.extend(params)
        if in_span:
            yield "</span>"
            in_span = False

        if not ansi_codes:
            continue

        style = [
            "; ".join([f"{k}: {v}" for k, v in styles[k].items()]).strip()
            for k in ansi_codes
            if k in styles
        ]
        yield '<span style="%s">' % "; ".join(style)

        in_span = True

    yield ansi_string[previous_end:]
    if in_span:
        yield "</span>"
        in_span = False
