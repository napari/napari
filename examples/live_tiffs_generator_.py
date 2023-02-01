"""
Live tiffs generator
====================

Simulation of microscope acquisition. This code generates time series tiffs in
an output directory (must be supplied by the user).

.. tags:: experimental
"""


import argparse
import os
import sys
import time

import numpy as np
import tifffile
from skimage import data

parser = argparse.ArgumentParser()
parser.add_argument('outdir', help='output directory for tiffs')
parser.add_argument(
    '--sleep-time',
    help='how long to sleep between volumes, in seconds',
    type=float,
    default=1.0,
)
parser.add_argument(
    '-n', help='total number of volumes', type=int, default=100
)


def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    outdir = args.outdir
    sleep_time = args.sleep_time
    n = args.n
    fractions = np.linspace(0.05, 0.5, n)
    os.makedirs(outdir, exist_ok=True)
    for i, f in enumerate(fractions):
        # We are using skimage binary_blobs which generate's synthetic binary
        # image with several rounded blob-like objects and write them into files.
        curr_vol = 255 * data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        ).astype(np.uint8)
        tifffile.imwrite(
            os.path.join(outdir, f'{i}.tiff'), curr_vol, compress=6
        )
        time.sleep(sleep_time)
    # create a final.log file as an indicator for end of acquisition
    final_file = open(os.path.join(outdir, 'final.log'), 'w')
    final_file.close()


if __name__ == '__main__':
    main()
