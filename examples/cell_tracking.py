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

# Extract silver truth data for trackastra
st_url = "doi:10.5281/zenodo.15852284/masks_pred.npz"
st_path = pooch.retrieve(
    url=st_url,
    fname="masks_pred.npz",
    known_hash=None,
    path=pooch.os_cache("napari cell tracking example")
)

# Extract raw tif files for trackastra
tif_url = "doi:10.5281/zenodo.17643282/01(1).zip"
tif_file = pooch.retrieve(
    url=tif_url,
    fname="01(1).zip",
    known_hash=None,
    path=pooch.os_cache("napari_cell_tracking_example"),
    processor=pooch.Unzip(),
)

# Load the downloaded masks
masks_npz = np.load(st_path)
masks = masks_npz['masks']

# Sort through images and stack them
imgs = sorted(tif_file)

images = np.stack([
        tifffile.imread(fn) for fn in imgs
        ])

# Initiate trackastra model
model = Trackastra.from_pretrained("general_2d", device='cpu')

# Generate tracks
graph, *_ = model.track(imgs=images, masks=masks, mode='greedy')
tracks_df, tracked_masks = graph_to_ctc(graph=graph, masks_original=masks)

napari_tracks, napari_graph = ctc_to_napari_tracks(segmentation=tracked_masks, man_track=tracks_df)

# Add Napari viewer
viewer = napari.Viewer()
viewer.add_image(images, name='Images')
viewer.add_labels(masks, name='Predicted Masks')
viewer.add_tracks(napari_tracks, graph=napari_graph, name="Tracks")

if __name__ == "__main__":
    napari.run()
