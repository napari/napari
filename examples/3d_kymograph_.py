"""
3D Kymographs
=============

This example demonstrates that the volume rendering capabilities of napari
can also be used to render 2d timelapse acquisitions as kymographs.

.. tags:: experimental
"""
from itertools import product

import numpy as np
from tqdm import tqdm

import napari

try:
    from omero.gateway import BlitzGateway
except ModuleNotFoundError:
    print("Could not import BlitzGateway which is")
    print("required to download the sample datasets.")
    print("Please install omero-py:")
    print("https://pypi.org/project/omero-py/")
    exit(-1)


def IDR_fetch_image(image_id: int, progressbar: bool = True) -> np.ndarray:
    """
    Download the image with id image_id from the IDR

    Will fetch all image planes corresponding to separate
    timepoints/channels/z-slices and return a numpy
    array with dimension order (t,z,y,x,c)

    Displaying download progress can be disabled by passing
    False to progressbar.
    """

    conn = BlitzGateway(
        host="ws://idr.openmicroscopy.org/omero-ws",
        username="public",
        passwd="public",
        secure=True,
    )
    conn.connect()
    conn.c.enableKeepAlive(60)

    idr_img = conn.getObject("Image", image_id)
    idr_pixels = idr_img.getPrimaryPixels()

    _ = idr_img
    nt, nz, ny, nx, nc = (
        _.getSizeT(),
        _.getSizeZ(),
        _.getSizeY(),
        _.getSizeX(),
        _.getSizeC(),
    )

    plane_indices = list(product(range(nz), range(nc), range(nt)))
    idr_plane_iterator = idr_pixels.getPlanes(plane_indices)

    if progressbar:
        idr_plane_iterator = tqdm(idr_plane_iterator, total=len(plane_indices))

    _tmp = np.asarray(list(idr_plane_iterator))
    _tmp = _tmp.reshape((nz, nc, nt, ny, nx))
    # the following line reorders the axes (no summing, despite the name)
    return np.einsum("jmikl", _tmp)


description = """
3D-Kymographs in Napari
=======================

About
=====
This example demonstrates that the volume rendering capabilities of napari
can also be used to render 2d timelapse acquisitions as kymographs.
Kymographs, also called space-time images, are a powerful tool to visualize
the dynamics of processes.
The most common way to visualize kymographs is to pick a single line through
a 2D image and visualize the time domain along a second axes.
Napari is not limited to 2D visualization an by harnessing its volume
volume rendering capabilities, we can create a 3D kymograph,
a powerful visualization that provides an overview of the complete
spatial and temporal data from a single view.

Using napari's grid mode we can juxtapose multiple such 3D kymographs to
highlight the differences in cell dynamics under different siRNA treatments.

The selected samples are from the Mitocheck screen and demonstrate siRNA
knockdowns of several genes.
The date is timelapse fluorescence microscopy of HeLa cells, with GFP-
tagged histone revealing the chromosomes.

In the juxtaposed kymographs the reduced branching for the mitotitic
phenotypes caused by INCENP, AURKB and KIF11 knockdown compared to
TMPRSS11A knockdown is immediately obvious.

Data Source
===========
The samples to demonstrate this is downloaded from IDR:
https://idr.openmicroscopy.org/webclient/?show=screen-1302

Reference
=========
The data comes from the Mitocheck screen:

Phenotypic profiling of the human genome by time-lapse microscopy reveals cell
division genes.

Neumann B, Walter T, Hériché JK, Bulkescher J, Erfle H, Conrad C, Rogers P,
Poser I, Held M, Liebel U, Cetin C, Sieckmann F, Pau G, Kabbe R, Wünsche A,
Satagopam V, Schmitz MH, Chapuis C, Gerlich DW, Schneider R, Eils R, Huber W,
Peters JM, Hyman AA, Durbin R, Pepperkok R, Ellenberg J.
Nature. 2010 Apr 1;464(7289):721-7.
doi: 10.1038/nature08869.

Acknowledgements
================
Beate Neumann (EMBL) for helpful advice on mitotic phenotypes.

"""

print(description)

samples = (
    {"IDRid": 2864587, "description": "AURKB knockdown", "vol": None},
    {"IDRid": 2862565, "description": "KIF11 knockdown", "vol": None},
    {"IDRid": 2867896, "description": "INCENP knockdown", "vol": None},
    {"IDRid": 1486532, "description": "TMPRSS11A knockdown", "vol": None},
)

print("-------------------------------------------------------")
print("Sample datasets will require ~490 MB download from IDR.")
answer = input("Press Enter to proceed, 'n' to cancel: ")
if answer.lower().startswith('n'):
    print("User cancelled download. Exiting.")
    exit(0)
print("-------------------------------------------------------")
for s in samples:
    print(f"Downloading sample {s['IDRid']}.")
    print(f"Description: {s['description']}")
    s["vol"] = np.squeeze(IDR_fetch_image(s["IDRid"]))

v = napari.Viewer(ndisplay=3)
scale = (5, 1, 1)  # "stretch" time domain
for s in samples:
    v.add_image(
        s["vol"], name=s['description'], scale=scale, blending="opaque"
    )

v.grid.enabled = True  # show the volumes in grid mode
v.axes.visible = True  # magenta error shows time direction

# set an oblique view angle onto the kymograph grid
v.camera.center = (440, 880, 1490)
v.camera.angles = (-20, 23, -50)
v.camera.zoom = 0.17

napari.run()