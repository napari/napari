"""
napari command line viewer.
"""
import argparse
import sys

from . import __version__, gui_qt, view_path
from .utils import citation_text, sys_info


class InfoAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        print(sys_info())
        sys.exit()


class CitationAction(argparse.Action):
    def __call__(self, *args, **kwargs):
        print(citation_text)
        sys.exit()


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('images', nargs='*', help='Images to view.')
    parser.add_argument(
        '-v',
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
        help='Concatenate multiple input files into a single stack.',
    )
    parser.add_argument(
        '-r',
        '--rgb',
        help='Treat images as RGB.',
        action='store_true',
        default=None,
    )
    parser.add_argument(
        '-g',
        '--grayscale',
        dest='rgb',
        action='store_false',
        help='interpret all dimensions in the image as spatial',
    )
    args = parser.parse_args()
    with gui_qt(startup_logo=True):
        view_path(args.images, stack=args.stack)


if __name__ == '__main__':
    sys.exit(main())
