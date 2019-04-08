"""
napari command line viewer.
"""
import argparse
import numpy as np
from skimage import io

from .util import app_context
from . import ViewerApp


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('images', nargs='+', help='Images to view.')
    parser.add_argument('--layers', action='store_true',
                        help='Treat multiple input images as layers.')
    parser.add_argument('-m', '--multichannel', help='Treat images as RGB.',
                        action='store_true')
    args = parser.parse_args()
    with app_context():
        images = io.ImageCollection(args.images, conserve_memory=False)
        if args.layers:
            ViewerApp(*images, multichannel=args.multichannel)
        else:
            # TODO: change to stack axis=0 when dims fixed
            image = np.stack(images, axis=-2 if args.multichannel else -1)
            ViewerApp(image, multichannel=args.multichannel)