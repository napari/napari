"""
Display a surface timeseries using data from nilearn
"""

try:
    from nilearn import datasets
    from nilearn import surface
except ImportError:
    raise ImportError("""This example uses data and methods from nilearn but
    nilearn installed. To install try 'pip install nilearn'.""")

import numpy as np
import napari


# Fetch datasets - this will download dataset if datasets are not found
nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)
fsaverage = datasets.fetch_surf_fsaverage()

# Load surface data and resting state time series from nilearn
pial_left = surface.load_surf_data(fsaverage['pial_left'])
sulc_left = surface.load_surf_data(fsaverage['sulc_left'])
timeseries = surface.load_surf_data(nki_dataset['func_left'][0]).transpose((1, 0))

with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=3)

    # add the mri
    viewer.add_surface((pial_left[0], pial_left[1], sulc_left), name='base')
    viewer.add_surface((pial_left[0], pial_left[1], timeseries),
                       colormap='turbo', opacity=0.9,
                       contrast_limits=[-1.5, 3.5], name='timeseries')
