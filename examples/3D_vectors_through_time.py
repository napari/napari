"""
3D vector field and image evolving through time
================================================

This example is from a mechanical test on a granular material, where multiple 3D volume have been acquired using x-ray tomography during axial loading.
We have used (spam)[https://www.spam-project.dev/] to label individual particles (with a watershed + tidying up manually), and these particles are tracked through time with 3D "discrete" volume correlation *incrementally*:
t0 → t1 and then t1 → t2.
Although we're also measuring rotations (and strains), but here we're interested to visualise the displacement field on top of the image to spot tracking errors.



.. tags:: visualization-nD
"""

import pooch
import tifffile
import numpy
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

greyFiles = sorted(pooch.retrieve(
    "doi:10.5281/zenodo.17668709/grey.zip",
    known_hash="md5:760be2bad68366872111410776563760",
    processor=pooch.Unzip(),
    progressbar=True
))

# Load individual 3D images as a 4D with a list comprehension, skipping last one
# result is a T, Z, Y, X 16-bit array
greys = numpy.array([tifffile.imread(greyFile) for greyFile in greyFiles[0:-1]])

# load incremental TSV tracking files from spam-ddic, [::2] is to skip VTK files also in folder
trackingFiles = sorted(pooch.retrieve(
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

for t, trackingFile in enumerate(trackingFiles):
    # load the indicator for convergence
    returnStatus = numpy.genfromtxt(trackingFile, skip_header=1, usecols=(19))

    # Load coords and displacements, keeping only converged results (returnStatus==2)
    coords = numpy.genfromtxt(trackingFile, skip_header=1, usecols=(1,2,3))[returnStatus==2]
    disps = numpy.genfromtxt(trackingFile, skip_header=1, usecols=(4,5,6))[returnStatus==2]

    # Compute lengths in order to colour vectors
    lengths = numpy.linalg.norm(disps, axis=1)

    # Add an extra dimension to coordinates to place them in time, and fill it with t
    coords = numpy.hstack([numpy.ones((coords.shape[0],1))*t, coords])
    # Add an extra time dimension of zeros for the "end" of the vector
    disps = numpy.hstack([numpy.zeros((disps.shape[0],1)), disps])

    # Add to lists
    coords_all.append(coords)
    disps_all.append(disps)
    lengths_all.append(lengths)

# Flatten and make arrays
coords_all = numpy.concatenate(coords_all)
disps_all = numpy.concatenate(disps_all)
lengths_all = numpy.concatenate(lengths_all)


v = napari.Viewer(ndisplay=3)

v.add_image(
    greys,
    contrast_limits=[15000, 50000],
    rendering="attenuated_mip",
    attenuation=0.333
)

v.add_vectors(
  numpy.stack([coords_all, disps_all], axis=1),
  vector_style='arrow',
  length=1,
  properties={'disp_norm': lengths_all},
  edge_colormap='plasma',
  edge_width=3,
  out_of_slice_display=True,
)

v.camera.angles = (2,-11,-23.5)
v.camera.orientation = ('away','up','right')
v.dims.current_step = (11,199, 124, 124)


if __name__ == '__main__':
    napari.run()
