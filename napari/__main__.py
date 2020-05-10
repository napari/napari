"""
napari command line viewer.
"""
import argparse
import logging
import sys
from ast import literal_eval
from typing import Any, Dict, List

from . import __version__, gui_qt, view_path
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
    from .components.add_layers_mixin import valid_add_kwargs

    out: Dict[str, Any] = dict()
    valid = set.union(*valid_add_kwargs().values())
    for i, arg in enumerate(unknown):
        if not arg.startswith("--"):
            continue
        if "=" in arg:
            sys.exit(f"error: '=' in argument {arg}. (Use space instead)")
        key = arg[2:].replace("-", "_")
        if key not in valid:
            sys.exit(f"error: unrecognized arguments: {arg}")
        try:
            next_arg = unknown[i + 1]
            if next_arg.startswith("--"):
                raise IndexError()
        except IndexError:
            sys.exit(f"error: argument {arg} expected one argument")
        try:
            val = literal_eval(next_arg)
        except Exception:
            val = next_arg
        out[key] = val
    return out


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('images', nargs='*', help='Images to view.')
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help="increase output verbosity",
    )
    parser.add_argument(
        '--version', action='version', version=f'napari version {__version__}',
    )
    parser.add_argument(
        '--plugin', help='specify plugin name when opening a file',
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
        help='Concatenate multiple input files into a single stack.',
    )

    args, unknown = parser.parse_known_args()
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
        if not args.images:
            sys.exit(
                "error: The '--plugin' argument is only valid "
                "when providing a file name"
            )
        # I *think* that Qt is looking in sys.argv for a flag `--plugins`,
        # which emits "WARNING: No such plugin for spec 'builtins'"
        # so remove --plugin from sys.argv to prevent that warningz
        sys.argv.remove('--plugin')

    with gui_qt(startup_logo=True):
        view_path(args.images, stack=args.stack, plugin=args.plugin, **kwargs)


if __name__ == '__main__':
    sys.exit(main())
