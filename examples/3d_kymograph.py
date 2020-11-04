from typing import Dict, List, Tuple
import numpy as np
import napari
from tqdm import tqdm
from omero.gateway import BlitzGateway
from itertools import product

# Helper function to retrieve sample time-lapse movies
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
    return np.einsum("jmikl", _tmp)


# Example:


description = """
3D-Kymographs in Napari

This example demonstrates that the volume rendering capabilities of napari can also
be used to render 2d timelapse acquisitions as kymographs. Kymographs, also called 
space-time images, can be a powerful tool to visualize dynamics of processes. However, 
the most common way to visualize kymographs is to pick a single line through a 2D image
and visualize the time domain as a second axes. By using volume rendering instead, we can 
create a visualization that gives an overview of the complete spatial and time course
from a single view.

The sample data to demonstrate this is downloaded from IDR:
https://idr.openmicroscopy.org/webclient/?show=screen-1302

and comes from the Mitocheck screen:

Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes. 

Neumann B, Walter T, Hériché JK, Bulkescher J, Erfle H, Conrad C, Rogers P, Poser I, Held M, 
Liebel U, Cetin C, Sieckmann F, Pau G, Kabbe R, Wünsche A, Satagopam V, Schmitz MH, Chapuis C,
Gerlich DW, Schneider R, Eils R, Huber W, Peters JM, Hyman AA, Durbin R, Pepperkok R, Ellenberg J.
Nature. 2010 Apr 1;464(7289):721-7. 
doi: 10.1038/nature08869. 

"""

print(description)

samples = (
    {"IDRid": 1486532, "description": "??? knockdown", "vol": None},
    {"IDRid": 2862565, "description": "KIF11 knockdown", "vol": None},
)

print("---------------------")
for s in samples:
    print(f"Downloading sample {s['IDRid']}.")
    print(f"Description: {s['description']}")
    s["vol"] = np.squeeze(IDR_fetch_image(s["IDRid"]))

with napari.gui_qt():
    v = napari.Viewer(ndisplay=3)
    scale = (5, 1, 1)  # "stretch" time domain
    for s in samples:
        v.add_image(
            s["vol"], name=s['description'], scale=scale, blending="opaque"
        )
    # oblique view angle onto the kymograph
    v.camera.center = (230, 510, 670)
    v.camera.angles = (-20, 30, -50)
    v.camera.zoom = 0.3
