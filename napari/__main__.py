"""
napari command line viewer.
"""
import argparse
import logging
import os
import runpy
import sys
import warnings
from ast import literal_eval
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List

from .utils.translations import trans


class InfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --info"
        from npe2 import cli

        from napari.utils import sys_info

        logging.basicConfig(level=logging.WARNING)
        print(sys_info())
        print("Plugins:")
        cli.list(fields="", sort="0", format="compact")
        sys.exit()


class PluginInfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --info"
        logging.basicConfig(level=logging.WARNING)
        from npe2 import cli

        cli.list(
            fields="name,version,npe2,contributions",
            sort="name",
            format="table",
        )
        sys.exit()


class CitationAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --citation"
        from napari.utils import citation_text

        logging.basicConfig(level=logging.WARNING)
        print(citation_text)
        sys.exit()


def validate_unknown_args(unknown: List[str]) -> Dict[str, Any]:
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

    out: Dict[str, Any] = {}
    valid = set.union(*valid_add_kwargs().values())
    for i, arg in enumerate(unknown):
        if not arg.startswith("--"):
            continue

        if "=" in arg:
            key, value = arg.split("=", maxsplit=1)
        else:
            key = arg
        key = key.lstrip('-').replace("-", "_")

        if key not in valid:
            sys.exit(f"error: unrecognized arguments: {arg}")

        if "=" not in arg:
            try:
                value = unknown[i + 1]
                if value.startswith("--"):
                    raise IndexError()
            except IndexError:
                sys.exit(f"error: argument {arg} expected one argument")
        try:
            value = literal_eval(value)
        except Exception:
            value = value

        out[key] = value
    return out


def parse_sys_argv():
    """Parse command line arguments."""

    from napari import __version__, layers
    from napari.components.viewer_model import valid_add_kwargs

    kwarg_options = []
    for layer_type, keys in valid_add_kwargs().items():
        kwarg_options.append(f"  {layer_type.title()}:")
        keys = {k.replace('_', '-') for k in keys}
        lines = wrap(", ".join(sorted(keys)), break_on_hyphens=False)
        kwarg_options.extend([f"    {line}" for line in lines])

    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="optional layer-type-specific arguments (precede with '--'):\n"
        + "\n".join(kwarg_options),
    )
    parser.add_argument('paths', nargs='*', help='path(s) to view.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help="increase output verbosity",
    )
    parser.add_argument(
        '-w',
        '--with',
        dest='with_',
        nargs='+',
        metavar=('PLUGIN_NAME', 'WIDGET_NAME'),
        help=(
            "open napari with dock widget from specified plugin name."
            "(If plugin provides multiple dock widgets, widget name must also "
            "be provided)"
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
        metavar="TYPE",
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
        if item.startswith("--"):
            unknown.append(args.paths.pop(len(args.paths) - idx - 1))
    kwargs = validate_unknown_args(unknown) if unknown else {}

    return args, kwargs


def _run():
    from napari import Viewer, run
    from napari.settings import get_settings

    """Main program."""
    args, kwargs = parse_sys_argv()

    # parse -v flags and set the appropriate logging level
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(2, args.verbose)]  # prevent index error
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt='%H:%M:%S',
    )

    if args.reset:
        if args.settings_path:
            settings = get_settings(path=args.settings_path)
        else:
            settings = get_settings()
        settings.reset()
        settings.save()
        sys.exit("Resetting settings to default values.\n")

    if args.plugin:
        # make sure plugin is only used when files are specified
        if not args.paths:
            sys.exit(
                "error: The '--plugin' argument is only valid "
                "when providing a file name"
            )
        # I *think* that Qt is looking in sys.argv for a flag `--plugins`,
        # which emits "WARNING: No such plugin for spec 'builtins'"
        # so remove --plugin from sys.argv to prevent that warningz
        sys.argv.remove('--plugin')

    if any(p.endswith('.py') for p in args.paths):
        # we're running a script
        if len(args.paths) > 1:
            sys.exit(
                'When providing a python script, only a '
                'single positional argument may be provided'
            )

        # run the file
        mod = runpy.run_path(args.paths[0])

        from napari_plugin_engine.markers import HookImplementationMarker

        # if this file had any hook implementations, register and run as plugin
        if any(isinstance(i, HookImplementationMarker) for i in mod.values()):
            _run_plugin_module(mod, os.path.basename(args.paths[0]))

    else:
        if args.with_:
            from .plugins import _initialize_plugins, _npe2, plugin_manager

            # if a plugin widget has been requested, this will fail immediately
            # if the requested plugin/widget is not available.
            _initialize_plugins()
            plugin_manager.discover_widgets()
            pname, *wnames = args.with_
            if wnames:
                for wname in wnames:
                    _npe2.get_widget_contribution(
                        pname, wname
                    ) or plugin_manager.get_widget(pname, wname)
            else:
                _npe2.get_widget_contribution(
                    pname
                ) or plugin_manager.get_widget(pname)

        from napari._qt.widgets.qt_splash_screen import NapariSplashScreen

        splash = NapariSplashScreen()
        splash.close()  # will close once event loop starts

        # viewer _must_  be kept around.
        # it will be referenced by the global window only
        # once napari has finished starting
        # but in the meantime if the garbage collector runs;
        # it will collect it and hang napari at start time.
        # in a way that is machine, os, time (and likely weather dependant).
        viewer = Viewer()

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
        viewer._window._qt_viewer._qt_open(
            args.paths,
            stack=args.stack,
            plugin=args.plugin,
            layer_type=args.layer_type,
            **kwargs,
        )

        if args.with_:
            pname, *wnames = args.with_
            if wnames:
                for wname in wnames:
                    viewer.window.add_plugin_dock_widget(pname, wname)
            else:
                viewer.window.add_plugin_dock_widget(pname)

        # only necessary in bundled app, but see #3596
        from napari.utils.misc import (
            install_certifi_opener,
            running_as_bundled_app,
        )

        if running_as_bundled_app():
            install_certifi_opener()
        run(gui_exceptions=True)


def _run_plugin_module(mod, plugin_name):
    """Register `mod` as a plugin, find/create viewer, and run napari."""
    from napari import Viewer, run
    from napari.plugins import plugin_manager

    plugin_manager.register(mod, name=plugin_name)

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

    # finally, if the file declared a dock widget, add it to the viewer.
    dws = plugin_manager.hooks.napari_experimental_provide_dock_widget
    if any(i.plugin_name == plugin_name for i in dws.get_hookimpls()):
        _v.window.add_plugin_dock_widget(plugin_name)

    run()


def _run_pythonw(python_path):
    """Execute this script again through pythonw.

    This can be used to ensure we're using a framework
    build of Python on macOS, which fixes frozen menubar issues.

    Parameters
    ----------
    python_path : pathlib.Path
        Path to python framework build.
    """
    import subprocess

    cwd = Path.cwd()
    cmd = [python_path, '-m', 'napari']
    env = os.environ.copy()

    # Append command line arguments.
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    result = subprocess.run(cmd, env=env, cwd=cwd)
    sys.exit(result.returncode)


def _run_as_napari_app():
    """
    Fixes the app name on MacOS menu bar
    See https://github.com/napari/napari/issues/4910
    for details
    """
    if "_RUNNING_AS_NAPARI_APP" in os.environ:
        # This function already ran, do not recurse!
        # We also restore sys.executable to its initial value
        if exe := os.environ.pop("_NAPARI_OLD_SYS_EXECUTABLE", ""):
            sys.executable = exe
        return

    if os.path.basename(sys.executable) == "napari":
        # If this is true, macOS should have picked the right name
        # and we do not need workarounds
        return

    import subprocess
    from tempfile import TemporaryDirectory

    cwd = Path.cwd()
    env = os.environ.copy()
    # Sentinel var to prevent a recursion
    env["_RUNNING_AS_NAPARI_APP"] = "1"
    # Pass original sys.executable to the subprocess so it can restore it later
    env["_NAPARI_OLD_SYS_EXECUTABLE"] = sys.executable
    with TemporaryDirectory(prefix="symlink-to-fix-macos-menu-name-") as tmp:
        # By using a symlink with basename napari
        # we make macOS take 'napari' as the program name
        napari_link = os.path.join(tmp, "napari")
        os.symlink(sys.executable, napari_link)
        cmd = [napari_link, '-m', 'napari']
        # Append command line arguments.
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])

        result = subprocess.run(cmd, env=env, cwd=cwd)
        sys.exit(result.returncode)


def main():
    # Ensure we're always using a "framework build" on the latest
    # macOS to ensure menubar works without needing to refocus napari.
    # We try this for macOS later than the Catelina release
    # See https://github.com/napari/napari/pull/1554 and
    # https://github.com/napari/napari/issues/380#issuecomment-659656775
    # and https://github.com/ContinuumIO/anaconda-issues/issues/199
    import platform

    from qtpy import API_NAME

    _MACOS_AT_LEAST_CATALINA = (
        sys.platform == "darwin"
        and int(platform.release().split('.')[0]) >= 19
    )
    _MACOS_AT_LEAST_BIG_SUR = (
        sys.platform == "darwin"
        and int(platform.release().split('.')[0]) >= 20
    )

    _RUNNING_CONDA = "CONDA_PREFIX" in os.environ
    _RUNNING_PYTHONW = "PYTHONEXECUTABLE" in os.environ

    # quick fix for Big Sur py3.9 and qt 5
    if _MACOS_AT_LEAST_BIG_SUR and '6' not in API_NAME:
        os.environ['QT_MAC_WANTS_LAYER'] = '1'

    if (
        _MACOS_AT_LEAST_CATALINA
        and not _MACOS_AT_LEAST_BIG_SUR
        and _RUNNING_CONDA
        and not _RUNNING_PYTHONW
    ):
        python_path = Path(sys.exec_prefix) / 'bin' / 'pythonw'

        if python_path.exists():
            # Running again with pythonw will exit this script
            # and use the framework build of python.
            _run_pythonw(python_path)
        else:
            msg = (
                'pythonw executable not found.\n'
                'To unfreeze the menubar on macOS, '
                'click away from napari to another app, '
                'then reactivate napari. To avoid this problem, '
                'please install python.app in conda using:\n'
                'conda install -c conda-forge python.app'
            )
            warnings.warn(msg)

    if sys.platform == "darwin":
        # Prevent https://github.com/napari/napari/issues/3415
        import multiprocessing

        multiprocessing.set_start_method('fork')

        # Make sure menu bar in macOS uses 'napari' as the app name
        if "napari" not in os.environ.get("__CFBundleIdentifier", ""):
            # When napari is launched from the bundle shortcut
            # it already has the right 'napari' name in the app title
            # and __CFBundleIdentifier is set to 'com.napari._(<version>)'
            _run_as_napari_app()

    _run()


if __name__ == '__main__':
    sys.exit(main())
