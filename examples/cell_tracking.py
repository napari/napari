"""
Cell tracking example
=====================

This example demonstrates how the track layer is used
for visualizing cell tracking data, by displaying 2D + time dataset
of cells with colored properties for track-id.

Thanks to Dr. Alessia Ruggieri and Philipp Klein, Centre for Integrative Infectious
Disease Research (CIID), University Hospital Heidelberg, Germany for the data.
You can find the data on: https://doi.org/10.5281/zenodo.15597019

.. tags:: visualization-advanced, colored 2D tracks
"""
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pooch
import tifffile
from trackastra.model import Trackastra
from trackastra.tracking import ctc_to_napari_tracks, graph_to_ctc

import napari

# Create temporary directory
tmp_dir = Path(pooch.os_cache('napari-cell-tracking-example'))
tmp_dir.mkdir(parents=True, exist_ok=True)

#Download ground truth data
gt_url = 'https://zenodo.org/records/16884151/files/TRA.zip?download=1'
gt_path = pooch.retrieve(url=gt_url,
                         fname="TRA.zip",
                         known_hash=None,
                         path=pooch.os_cache("cell_tracking")
                         )
# Make ground truth sub directory
gt_folder = tmp_dir / "TRA"

# Extract compressed ground truth data
if not gt_folder.exists():
    print(f"Extracting {gt_path} to {gt_folder}")
    with zipfile.ZipFile(gt_path, 'r') as zip_ref:
        zip_ref.extractall(gt_folder)
else:
    print(f"Using cached ground truth at {gt_folder}")

# Extract data for trackastra based silver truth
st_url = "https://zenodo.org/records/15852284/files/masks_pred.npz?download=1"
st_path = pooch.retrieve(
    url=st_url,
    fname="masks_pred.npz",
    known_hash=None,
    path=pooch.os_cache("napari cell tracking example")
)

# Extract raw tif files for trackastra
tif_url = "https://zenodo.org/records/17643282/files/01(1).zip?download=1"
tif_path = pooch.retrieve(
    url=tif_url,
    fname="01(1).zip",
    known_hash=None,
    path=pooch.os_cache("napari_cell_tracking_example")
)

# Load the downloaded masks
masks_npz = np.load(st_path)
masks = masks_npz['masks']
print(f"mask shape is {masks.shape}")

# Make raw tif sub directory
tif_folder = tmp_dir/ "01"

# Extract raw images
with zipfile.ZipFile(tif_path, 'r') as zip: 
    zip.extractall(path=tif_folder)

# Sort through images and stack them
imgs = [f for f in sorted((tif_folder).rglob ("*.tif"))
        if not f.name.startswith("._")]

valid_files = []

for f in imgs:
    try:
        with tifffile.TiffFile(f) as tif:
            tif.pages[0]
        valid_files.append(f)
    except Exception as e:
        print(f"Skipping {f.name} — not a valid TIFF ({e})")


images = np.stack([
        tifffile.imread(fn) for fn in sorted(imgs)
        ])
print(f"image shape is {images.shape}")

# Make sure mask shape and image shape is the same
if images.shape[0] != masks.shape[0]:
    raise ValueError("Number of frames in imgs and masks differ")

if len(masks_npz.files) > 0:
    first_key = masks_npz.files[0]
    masks = masks_npz['masks']
    print(f"Loaded masks from key '{first_key}' with shape: {masks.shape}")
else:
    raise ValueError("No arrays found in npz file")

# Initiate trackastra model
model = Trackastra.from_pretrained("general_2d", device='cpu')

# Generate tracks
graph, *_ = model.track(imgs=images, masks=masks, mode='greedy')
tracks_df, tracked_masks = graph_to_ctc(graph=graph, masks_original=masks)

napari_tracks, napari_graph = ctc_to_napari_tracks(segmentation=tracked_masks, man_track=tracks_df)

# Quick check to see if everything looks ok
print(f"DataFrame columns: {tracks_df.columns.tolist()}")
print(f"DataFrame shape: {tracks_df.shape}")
print(f"First few rows:\n{tracks_df.head()}")

# Add Napari viewer
viewer = napari.Viewer()

# I am not sure if we should also open ground truth data here
#viewer.open(str(gt_folder), plugin='napari-ctc')

viewer.add_labels(masks, name='Predicted Masks')
viewer.add_tracks(napari_tracks, graph=napari_graph, name="Tracks")

napari.run()
