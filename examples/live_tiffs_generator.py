"""
Display one 4-D image layer using the add_image API
"""
import os
import sys
import time
import argparse

import numpy as np
from skimage import data
import tifffile


parser = argparse.ArgumentParser()
parser.add_argument('outdir', help='output directory for tiffs')
parser.add_argument('--sleep-time', help='how long to sleep between volumes, in seconds', default=1)
parser.add_argument('-n', help='total number of volumes', default=100)


def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    outdir = args.outdir
    sleep_time = args.sleep_time
    n = args.n
    fractions = np.linspace(0.05, 0.5, n)
    os.makedirs(outdir, exist_ok=True)
    for i, f in enumerate(fractions):
        curr_vol = 255 * data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        ).astype(np.uint8)
        tifffile.imwrite(os.path.join(outdir, f'{i}.tiff'), curr_vol, compress=6)
        time.sleep(sleep_time)

if __name__ == '__main__':
    main()