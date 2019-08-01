"""
napari command line viewer.
"""
import argparse
import numpy as np
from dask import delayed
import dask.array as da
from skimage import io

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
        action='store_true',
        help='Treat images as RGB.',
    )
    parser.add_argument(
        '-d',
        '--dask',
        action='store_true',
        help='Load images lazily when given multiple images. Ignored with '
        '--layers.',
    )
    parser.add_argument(
        '-c',
        '--clim-range',
        type=lambda s: tuple(map(float, s.split(','))),
        default=None,
        help='Define the contrast limits for the input image. Highly '
        'recommended when using --dask.',
    )
    args = parser.parse_args()
    with gui_qt():
        v = Viewer()
        images = io.ImageCollection(args.images, conserve_memory=True)
        if args.layers:
            for image in images:
                v.add_image(image, multichannel=args.multichannel)
        else:
            if len(images) > 0:
                if len(images) == 1:
                    image = images[0]
                else:
                    if args.dask:
                        i0 = images[0]
                        images_dask = [
                            da.from_delayed(
                                delayed(images.__getitem__)(i),
                                shape=i0.shape,
                                dtype=i0.dtype,
                            )
                            for i in range(len(images))
                        ]
                        image = da.stack(images_dask, axis=0)
                    else:
                        image = np.stack(images, axis=0)
                v.add_image(
                    image,
                    multichannel=args.multichannel,
                    clim_range=args.clim_range,
                )
