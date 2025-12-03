"""
3D vector field and image across time
=====================================

This example is from a mechanical test on a granular material, where multiple 3D volume have been acquired using x-ray tomography during axial loading.
We have used [`spam`](https://www.spam-project.dev/) to label individual particles (with watershed and some tidying up manually), and these particles are tracked through time with 3D "discrete" volume correlation *incrementally*:
t0 → t1 and then t1 → t2.
Although we're also measuring rotations (and strains), but here we're interested to visualise the displacement field on top of the image to spot tracking errors.

.. tags:: visualization-nD
"""

import numpy as np
import pooch
import tifffile

import napari

###############################################################################
# Input data
# -----------
#
# Input data are therefore:
#   - A series of 3D greyscale images
#   - A series of measured transformations between subsequent pairs of images
#   - We also have consistent labels through time which could also be visualised but are not here
#
# Let's download it!

grey_files = sorted(pooch.retrieve(
    "doi:10.5281/zenodo.17668709/grey.zip",
    known_hash="md5:760be2bad68366872111410776563760",
    processor=pooch.Unzip(),
    progressbar=True
))

# Load individual 3D images as a 4D with a list comprehension, skipping last one
# result is a T, Z, Y, X 16-bit array
greys = np.array([tifffile.imread(grey_file) for grey_file in grey_files[0:-1]])

# load incremental TSV tracking files from spam-ddic, [::2] is to skip VTK files also in folder
tracking_files = sorted(pooch.retrieve(
    "doi:10.5281/zenodo.17668709/ddic.zip",
    known_hash="md5:2d7c6a052f53b4a827ff4e4585644fac",
    processor=pooch.Unzip(),
    progressbar=True
))[::2]


###############################################################################
# Collect data together for napari
#----------------------------------
# We will loop through all the images in order to prepare the necessary data structure for napari
#
# These variables are going to contain coordinates, displacements and lengths (for colouring vectors) for all timesteps
coords_all = []
disps_all = []
lengths_all = []

for t, tracking_file in enumerate(tracking_files):
    # load the indicator for convergence
    returnStatus = np.genfromtxt(tracking_file, skip_header=1, usecols=(19))

    # Load coords and displacements, keeping only converged results (returnStatus==2)
    coords = np.genfromtxt(tracking_file, skip_header=1, usecols=(1,2,3))[returnStatus==2]
    disps = np.genfromtxt(tracking_file, skip_header=1, usecols=(4,5,6))[returnStatus==2]

    # Compute lengths in order to colour vectors
    lengths = np.linalg.norm(disps, axis=1)

    # Prepend an extra dimension to coordinates to place them in time, and fill it with the incremental t
    coords = np.hstack([np.ones((coords.shape[0],1))*t, coords])
    # Preprend zeros to the displacements (the "end" of the vector), since they do not displace through time
    disps = np.hstack([np.zeros((disps.shape[0],1)), disps])

    # Add to lists
    coords_all.append(coords)
    disps_all.append(disps)
    lengths_all.append(lengths)

# Concatenate into arrays
coords_all = np.concatenate(coords_all)
disps_all = np.concatenate(disps_all)
lengths_all = np.concatenate(lengths_all)
# Stack this into an array of size N (individual points) x 2 (vector start and length) x 4 (tzyx)
coords_displacements_all = np.stack([coords_all, disps_all], axis=1)

viewer = napari.Viewer(ndisplay=3)

viewer.add_image(
    greys,
    contrast_limits=[15000, 50000],
    rendering="attenuated_mip",
    attenuation=0.333
)

viewer.add_vectors(
  coords_displacements_all,
  vector_style='arrow',
  length=1,
  properties={'disp_norm': lengths_all},
  edge_colormap='plasma',
  edge_width=3,
  out_of_slice_display=True,
)

viewer.camera.angles = (2,-11,-23.5)
viewer.camera.orientation = ('away','up','right')
viewer.dims.current_step = (11,199, 124, 124)


if __name__ == '__main__':
    napari.run()
