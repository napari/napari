"""
napari command line viewer.
"""

import argparse
import contextlib
import logging
import sys
import warnings
from ast import literal_eval
from pathlib import Path
from textwrap import wrap
from typing import Any

from napari import Viewer
from napari.errors import ReaderPluginError
from napari.utils._startup_script import _run_configured_startup_script
from napari.utils.misc import maybe_patch_conda_exe
from napari.utils.translations import trans


class InfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --info"

        from napari.utils import sys_info

        logging.basicConfig(level=logging.WARNING)
        print(sys_info())  # noqa: T201
        sys.exit()


class PluginInfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --info"
        logging.basicConfig(level=logging.WARNING)
        from npe2 import cli

        cli.list(
            fields='name,version,npe2,contributions',
            sort='name',
            format='table',
        )
        sys.exit()


class CitationAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --citation"
        from napari.utils import citation_text

        logging.basicConfig(level=logging.WARNING)
        print(citation_text)  # noqa: T201
        sys.exit()


def validate_unknown_args(unknown: list[str]) -> dict[str, Any]:
    """Convert a list of strings into a dict of valid kwargs for add_* methods.

    Will exit program if any of the arguments are unrecognized, or are
    malformed.  Converts string to python type using literal_eval.

    Parameters
    ----------
    unknown : List[str]
        a list of strings gathered as "unknown" arguments in argparse.

    Returns
    -------
    kwargs : Dict[str, Any]
        {key: val} dict suitable for the viewer.add_* methods where ``val``
        is a ``literal_eval`` result, or string.
    """

    from napari.components.viewer_model import valid_add_kwargs

    out: dict[str, Any] = {}
    valid = set.union(*valid_add_kwargs().values())
    for i, raw_arg in enumerate(unknown):
        if not raw_arg.startswith('--'):
            continue
        arg = raw_arg.lstrip('-')

        key, *values = arg.split('=', maxsplit=1)
        key = key.replace('-', '_')
        if key not in valid:
            sys.exit(f'error: unrecognized argument: {raw_arg}')

        if values:
            value = values[0]
        else:
            if len(unknown) <= i + 1 or unknown[i + 1].startswith('--'):
                sys.exit(f'error: argument {raw_arg} expected one argument')
            value = unknown[i + 1]
        with contextlib.suppress(Exception):
            value = literal_eval(value)

        out[key] = value
    return out


def parse_sys_argv():
    """Parse command line arguments."""

    from napari import __version__, layers
    from napari.components.viewer_model import valid_add_kwargs

    kwarg_options = []
    for layer_type, keys in valid_add_kwargs().items():
        kwarg_options.append(f'  {layer_type.title()}:')
        keys = {k.replace('_', '-') for k in keys}
        lines = wrap(', '.join(sorted(keys)), break_on_hyphens=False)
        kwarg_options.extend([f'    {line}' for line in lines])

    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="optional layer-type-specific arguments (precede with '--'):\n"
        + '\n'.join(kwarg_options),
    )
    parser.add_argument('paths', nargs='*', help='path(s) to view.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='increase output verbosity',
    )
    parser.add_argument(
        '-w',
        '--with',
        dest='with_',
        nargs='+',
        action='append',
        default=[],
        metavar=('PLUGIN_NAME', 'WIDGET_NAME'),
        help=(
            'open napari with dock widget from specified plugin name.'
            '(If plugin provides multiple dock widgets, widget name must also '
            'be provided). Use __all__ to open all dock widgets of a '
            'specified plugin. Multiple widgets are opened in tabs.'
        ),
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'napari version {__version__}',
    )
    parser.add_argument(
        '--info',
        action=InfoAction,
        nargs=0,
        help='show system information and exit',
    )
    parser.add_argument(
        '--plugin-info',
        action=PluginInfoAction,
        nargs=0,
        help='show information about plugins and exit',
    )
    parser.add_argument(
        '--citation',
        action=CitationAction,
        nargs=0,
        help='show citation information and exit',
    )
    # Allow multiple --stack options to be provided.
    # Each stack option will result in its own stack
    parser.add_argument(
        '--stack',
        action='append',
        nargs='*',
        default=[],
        help='concatenate multiple input files into a single stack. Can be provided multiple times for multiple stacks.',
    )
    parser.add_argument(
        '--plugin',
        help='specify plugin name when opening a file',
    )
    parser.add_argument(
        '--layer-type',
        metavar='TYPE',
        choices=set(layers.NAMES),
        help=(
            'force file to be interpreted as a specific layer type. '
            f'one of {set(layers.NAMES)}'
        ),
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='reset settings to default values.',
    )
    parser.add_argument(
        '--settings-path',
        type=Path,
        help='use specific path to store and load settings.',
    )

    args, unknown = parser.parse_known_args()
    # this is a hack to allow using "=" as a key=value separator while also
    # allowing nargs='*' on the "paths" argument...
    for idx, item in enumerate(reversed(args.paths)):
        if item.startswith('--'):
            unknown.append(args.paths.pop(len(args.paths) - idx - 1))
    kwargs = validate_unknown_args(unknown) if unknown else {}

    return args, kwargs


def _run() -> None:
    from napari import run
    from napari.settings import get_settings

    """Main program."""
    args, kwargs = parse_sys_argv()

    # parse -v flags and set the appropriate logging level
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(2, args.verbose)]  # prevent index error
    logging.basicConfig(
        level=level,
        format='%(asctime)s : %(levelname)s : %(threadName)s : %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.reset:
        if args.settings_path:
            settings = get_settings(path=args.settings_path)
        else:
            settings = get_settings()
        settings.reset()
        settings.save()
        sys.exit('Resetting settings to default values.\n')

    if args.plugin:
        # make sure plugin is only used when files are specified
        if not args.paths:
            sys.exit(
                "error: The '--plugin' argument is only valid "
                'when providing a file name'
            )
        # I *think* that Qt is looking in sys.argv for a flag `--plugins`,
        # which emits "WARNING: No such plugin for spec 'builtins'"
        # so remove --plugin from sys.argv to prevent that warning
        sys.argv.remove('--plugin')

    else:
        if args.with_:
            from napari.plugins import (
                _initialize_plugins,
                _npe2,
            )

            # if a plugin widget has been requested, this will fail immediately
            # if the requested plugin/widget is not available.
            _initialize_plugins()

            npe2_plugins = []
            for plugin in args.with_:
                pname, *wnames = plugin
                for name, (w_pname, wnames) in _npe2.widget_iterator():
                    if name == 'dock' and pname == w_pname:
                        npe2_plugins.append(plugin)
                        if '__all__' in wnames:
                            wnames = wnames
                        break

                if wnames:
                    for wname in wnames:
                        _npe2.get_widget_contribution(pname, wname)
                else:
                    _npe2.get_widget_contribution(pname)

        # viewer _must_  be kept around.
        # it will be referenced by the global window only
        # once napari has finished starting
        # but in the meantime if the garbage collector runs;
        # it will collect it and hang napari at start time.
        # in a way that is machine, os, time (and likely weather dependant).
        viewer = Viewer()
        _run_configured_startup_script()

        # For backwards compatibility
        # If the --stack option is provided without additional arguments
        # just set stack to True similar to the previous store_true action
        if args.stack and len(args.stack) == 1 and len(args.stack[0]) == 0:
            warnings.warn(
                trans._(
                    "The usage of the --stack option as a boolean is deprecated. Please use '--stack file1 file2 .. fileN' instead. It is now also possible to specify multiple stacks of files to stack '--stack file1 file2 --stack file3 file4 file5 --stack ..'. This warning will become an error in version 0.5.0.",
                ),
                DeprecationWarning,
                stacklevel=3,
            )
            args.stack = True
        try:
            viewer._window._qt_viewer._qt_open(
                args.paths,
                stack=args.stack,
                plugin=args.plugin,
                layer_type=args.layer_type,
                **kwargs,
            )
        except ReaderPluginError:
            logging.getLogger('napari').exception(
                'Loading %s with %s failed with errors',
                args.paths,
                args.plugin,
            )

        if args.with_:
            for plugin in npe2_plugins:
                pname, *wnames = plugin
                if '__all__' in wnames:
                    for name, (
                        _pname,
                        wnames_collection,
                    ) in _npe2.widget_iterator():
                        if name == 'dock' and pname == _pname:
                            wnames = wnames_collection
                            break

                if wnames:
                    first_dock_widget = viewer.window.add_plugin_dock_widget(
                        pname, wnames[0], tabify=True
                    )[0]
                    for wname in wnames[1:]:
                        viewer.window.add_plugin_dock_widget(
                            pname, wname, tabify=True
                        )
                    first_dock_widget.show()
                    first_dock_widget.raise_()
                else:
                    viewer.window.add_plugin_dock_widget(pname, tabify=True)

        # only necessary in bundled app, but see #3596
        from napari.utils.misc import (
            install_certifi_opener,
            running_as_constructor_app,
        )

        if running_as_constructor_app():
            install_certifi_opener()
            maybe_patch_conda_exe()
        run(gui_exceptions=True)


def _run_plugin_module(mod, plugin_name):
    """Register `mod` as a plugin, find/create viewer, and run napari."""
    from napari import Viewer, run

    # now, check if a viewer was created, and if not, create one.
    for obj in mod.values():
        if isinstance(obj, Viewer):
            _v = obj
            break
    else:
        _v = Viewer()

    try:
        _v.window._qt_window.parent()
    except RuntimeError:
        # this script had a napari.run() in it, and the viewer has already been
        # used and cleaned up... if we eventually have "reusable viewers", we
        # can continue here
        return

    run()


def main():
    _run()


if __name__ == '__main__':
    sys.exit(main())
