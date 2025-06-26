"""
Updated Cell tracking example
==============================

This example demonstrates how the tracking layer is used
for displaying 2D + time dataset of cells with colored properties for track-id.

Thanks to Dr. Alessia Ruggieri and Philipp Klein, Centre for Integrative Infectious
Disease Research (CIID), University Hospital Heidelberg, Germany for the data.
You can find the data on: https://doi.org/10.5281/zenodo.15597019

"""
import zipfile
from pathlib import Path

import numpy as np
import pooch
import tifffile

import napari

###############################################################################
# Download the data
# ------------------
download = pooch.DOIDownloader(progressbar=True)
doi = "10.5281/zenodo.15597019/01.zip"
tmp_dir = Path(pooch.os_cache('napari-cell-tracking-example'))
#os.makedirs(tmp_dir, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

#data_files = "01.zip"

data_path = tmp_dir/"01.zip"

if not data_path.exists():
    print(f"downloading archive {data_path.name}")
    #download(f"doi:{doi}", output_file=data_path, path=data_files)
    download(f"doi:{doi}", output_file=data_path, pooch=None)
else:
    print(f"using cached {data_path}")

with zipfile.ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_dir)

print("Contents of tmp_dir after extraction:")
for p in tmp_dir.rglob("*"):
    print(p)

#imgs = glob.glob(os.path.join(tmp_dir, "*.tif"))
imgs = list((tmp_dir/"01").glob ("*.tif"))

if not imgs:
    raise FileNotFoundError("no tif files found")

images = np.stack([
        tifffile.imread(fn) for fn in sorted(imgs)
        ])
viewer = napari.Viewer()
viewer.add_image(images, name='Cell Tracking', colormap='gray',
                 metadata={'Unit': 'Hours'})

napari.run()
