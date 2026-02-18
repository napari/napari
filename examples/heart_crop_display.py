"""
Heart example
=============

Display an image with preset contrast limits and colormap, a segmentation mask and points with features.

This example demonstrates how to load and display a subset of channels from a multi-channel 2D image with preset contrast limits and colormap.
It adds a segmentation mask and points with features.

.. tags:: visualization-basic
"""

import pandas as pd
import tifffile

import napari

# read in image, segmentation mask and cell data
crop = tifffile.imread('/Users/margotchazotte/Documents/uni/PhD/napari_examples/heart_crop/image.tif')
crop_mask = tifffile.imread('/Users/margotchazotte/Documents/uni/PhD/napari_examples/heart_crop/segmentation.tif')
crop_cells = pd.read_csv('/Users/margotchazotte/Documents/uni/PhD/napari_examples/heart_crop/cell_data.csv')

# read in metadata containing contrast limits, colormap and channel index for each channel that should be included
channel_color_metadata = pd.read_csv('/Users/margotchazotte/Documents/uni/PhD/napari_examples/heart_crop/channel_color_metadata.csv')

# define colors for each cell type for the points layer
color_cycle = {
    'Cardiomyocytes': '#0000ff',
    'Cardiomyocytes Ankrd1+': "#b5791a",
    'Macrophages Trem2+':'#01f08a',
    'Fibroblasts': '#ff8000',
    'Endothelial cells': '#ffff00',
    'Mono / Macros Ccr2+':'#00ff00',
    'Neutrophils':'#ff00ff',
    'Smooth muscle cells':'#ff0000',
    'Endocardial cells': '#ffa700',
    'Other Leukocytes':"#7b22ff",
    'Macrophages Trem2-':"#ff0080"
    }

# subset cell features to those that should be added to the points layer
crop_feature = { 'label': crop_cells['label'],
           'cell_size': crop_cells['cell_size'],
           'cell_type': crop_cells['final_cell_type']}

# create napari viewer and add image channel by channel with contrast limits and colormap according to channel_color_metadata
viewer = napari.Viewer()
for idx, name in zip(channel_color_metadata['channel_index'], channel_color_metadata['name'], strict = True):
    viewer.add_image(
        crop[idx],
        name=name,
        rgb=False,
        contrast_limits=[channel_color_metadata['contrast_min'][channel_color_metadata['channel_index'] == idx].values[0],
                            channel_color_metadata['contrast_max'][channel_color_metadata['channel_index'] == idx].values[0]],
        blending='additive',
        colormap=channel_color_metadata['colorhex'][channel_color_metadata['channel_index'] == idx].values[0],
    )

# add segmetation mask, optionally add features which can be displayed by plugins
viewer.add_labels(crop_mask,
                  name='Segmentation mask',
                  #features=crop_feature
                  )

# add points with features, colored by cell type
viewer.add_points(
    crop_cells[['Y_centroid', 'X_centroid']].values,
    name='cell_types',
    features=crop_feature,
    face_color='cell_type',
    size=30,
    face_color_cycle=color_cycle
)
napari.run()
