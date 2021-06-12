"""
Display a surface timeseries using data from nilearn
"""

try:
    from nilearn import datasets
    from nilearn import surface
except ImportError:
    raise ImportError(
        "You must have nilearn installed to run this example."
    )

import numpy as np
import napari


# Fetch datasets - this will download dataset if datasets are not found
nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=1)
fsaverage = datasets.fetch_surf_fsaverage()

# Load surface data and resting state time series from nilearn
brain_vertices, brain_faces = surface.load_surf_data(fsaverage['pial_left'])
brain_vertex_depth = surface.load_surf_data(fsaverage['sulc_left'])
timeseries = surface.load_surf_data(nki_dataset['func_left'][0])
# nilearn provides data as n_vertices x n_timepoints, but napari requires the
# vertices axis to be placed last to match NumPy broadcasting rules
timeseries = timeseries.transpose((1, 0))

# create an empty viewer
viewer = napari.Viewer(ndisplay=3)

# add the mri
viewer.add_surface((brain_vertices, brain_faces, brain_vertex_depth), name='base')
viewer.add_surface((brain_vertices, brain_faces, timeseries),
                    colormap='turbo', opacity=0.9,
                    contrast_limits=[-1.5, 3.5], name='timeseries')

napari.run()