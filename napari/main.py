"""
napari command line viewer.
"""
import argparse
import numpy as np
from skimage import io
import os.path

from . import Viewer, gui_qt


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('images', nargs='*', help='Images to view.')
    parser.add_argument(
        '--layers',
        action='store_true',
        help='Treat multiple input images as layers.',
    )
    parser.add_argument(
        '-m',
        '--multichannel',
        help='Treat images as RGB.',
        action='store_true',
    )
    args = parser.parse_args()
    with gui_qt():
        v = Viewer()
        if len(args.images) == 0:
            return

        filename = args.images[0]
        if os.path.splitext(filename)[1] == '.napari':
            v.from_zarr(filename)
        elif os.path.splitext(filename[:-1])[1] == '.napari':
            v.from_zarr(filename[:-1])
        else:
            images = io.ImageCollection(args.images, conserve_memory=False)
            if args.layers:
                for image in images:
                    v.add_image(image, multichannel=args.multichannel)
            else:
                if len(images) > 0:
                    if len(images) == 1:
                        image = images[0]
                    else:
                        image = np.stack(images, axis=0)
                    v.add_image(image, multichannel=args.multichannel)
