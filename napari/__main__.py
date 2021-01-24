"""
napari command line viewer.
"""
import argparse
import logging
import os
import platform
import runpy
import sys
import warnings
from ast import literal_eval
from distutils.version import StrictVersion
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, List

from . import __version__, gui_qt, layers, view_path
from .components.viewer_model import valid_add_kwargs
from .utils import citation_text, sys_info


class InfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --info"
        logging.basicConfig(level=logging.WARNING)
        print(sys_info())
        sys.exit()


class CitationAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        # prevent unrelated INFO logs when doing "napari --citation"
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

    out: Dict[str, Any] = dict()
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


def _run():
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
        '--citation',
        action=CitationAction,
        nargs=0,
        help='show citation information and exit',
    )
    parser.add_argument(
        '--stack',
        action='store_true',
        help='concatenate multiple input files into a single stack.',
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

    args, unknown = parser.parse_known_args()
    # this is a hack to allow using "=" as a key=value separator while also
    # allowing nargs='*' on the "paths" argument...
    for idx, item in enumerate(reversed(args.paths)):
        if item.startswith("--"):
            unknown.append(args.paths.pop(len(args.paths) - idx - 1))
    kwargs = validate_unknown_args(unknown) if unknown else {}

    # parse -v flags and set the appropriate logging level
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(2, args.verbose)]  # prevent index error
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt='%H:%M:%S',
    )

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
        if len(args.paths) > 1:
            sys.exit(
                'When providing a python script, only a '
                'single positional argument may be provided'
            )

        with gui_qt(startup_logo=True) as app:
            if hasattr(app, '_splash_widget'):
                app._splash_widget.close()
            runpy.run_path(args.paths[0])
            if getattr(app, '_existed', False):
                sys.exit()
    else:
        with gui_qt(startup_logo=True, gui_exceptions=True):
            view_path(
                args.paths,
                stack=args.stack,
                plugin=args.plugin,
                layer_type=args.layer_type,
                **kwargs,
            )


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


def main():
    # Ensure we're always using a "framework build" on the latest
    # macOS to ensure menubar works without needing to refocus napari.
    # We try this for macOS later than the Catelina release
    # See https://github.com/napari/napari/pull/1554 and
    # https://github.com/napari/napari/issues/380#issuecomment-659656775
    # and https://github.com/ContinuumIO/anaconda-issues/issues/199
    _MACOS_AT_LEAST_CATALINA = sys.platform == "darwin" and StrictVersion(
        platform.release()
    ) > StrictVersion('19.0.0')
    _MACOS_AT_LEAST_BIG_SUR = sys.platform == "darwin" and StrictVersion(
        platform.release()
    ) > StrictVersion('20.0.0')

    _RUNNING_CONDA = "CONDA_PREFIX" in os.environ
    _RUNNING_PYTHONW = "PYTHONEXECUTABLE" in os.environ

    # quick fix for Big Sur py3.9
    if _MACOS_AT_LEAST_BIG_SUR:
        os.environ['QT_MAC_WANTS_LAYER'] = '1'

    if _MACOS_AT_LEAST_CATALINA and _RUNNING_CONDA and not _RUNNING_PYTHONW:
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
    _run()


if __name__ == '__main__':
    sys.exit(main())
