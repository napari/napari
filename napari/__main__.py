"""
napari command line viewer.
"""
import argparse
import sys

import numpy as np

from . import Viewer, __version__, gui_qt
from .utils import io, sys_info, citation_text


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
        '--layers',
        action='store_true',
        help='Treat multiple input images as layers.',
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
    parser.add_argument(
        '-D',
        '--use-dask',
        action='store_true',
        help='Use dask to read in images. This conserves memory. This option '
        'does nothing if a single image is given.',
        default=None,
    )
    parser.add_argument(
        '-N',
        '--use-numpy',
        action='store_false',
        dest='use_dask',
        help='Use NumPy to read in images. This can be more performant than '
        'dask if all the images fit in RAM. This option does nothing if '
        'only a single image is given.',
    )
    args = parser.parse_args()
    with gui_qt(startup_logo=True):
        v = Viewer()
        if len(args.images) > 0:
            images = io.magic_imread(
                args.images, use_dask=args.use_dask, stack=not args.layers
            )
            if args.layers:
                for layer in images:
                    if layer.dtype in (
                        np.int32,
                        np.uint32,
                        np.int64,
                        np.uint64,
                    ):
                        v.add_labels(layer)
                    else:
                        v.add_image(layer, rgb=args.rgb)
            else:
                v.add_image(images, rgb=args.rgb)


if __name__ == '__main__':
    sys.exit(main())
