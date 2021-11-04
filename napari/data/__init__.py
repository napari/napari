"""Demonstation datasets."""
import pooch
import tifffile

__all__ = [
    "cells3d",
    "mri",
    "plantseed",
]

registry = {
    "cells3d.tif": "afc7c7d80d38bfde09788b4064ac1e64ec14e88454ab785ebdc8dbba5ca3b222",
    "Haase_MRT_tfl3d1.tif": "md5:0253b9be403dd05cd38bea7cbcff00a5",
    "EM_C_6_c0.tif": "md5:6b346069c6620fd72ad90390e082a437",
}

registry_urls = {
    "cells3d.tif": "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif",
    "Haase_MRT_tfl3d1.tif": "https://zenodo.org/record/5090508/files/Haase_MRT_tfl3d1.tif?download=1",
    "EM_C_6_c0.tif": "https://zenodo.org/record/5090508/files/EM_C_6_c0.tif?download=1",
}


POOCH = pooch.create(
    # Pooch uses appdirs to select an appropriate directory for the cache
    # on each platform. https://github.com/ActiveState/appdirs
    # On linux this converges to '$HOME/.cache/napari'
    path=pooch.os_cache("napari"),
    base_url="",
    registry=registry,
    urls=registry_urls,
)


def cells3d():
    """3D fluorescence microscopy image of cells.

    The returned data is a 3D multichannel array with dimensions provided in
    ``(z, c, y, x)`` order. Each voxel has a size of ``(0.29 0.26 0.26)``
    micrometer. Channel 0 contains cell membranes, channel 1 contains nuclei.
    File size: 10.9 MB

    Returns
    -------
    cells3d: (60, 2, 256, 256) uint16 ndarray
        The volumetric images of cells taken with an optical microscope.

    Notes
    -----
    The data for this was provided by the Allen Institute for Cell Science.

    It has been downsampled by a factor of 4 in the row and column dimensions
    to reduce computational time.

    The microscope reports the following voxel spacing in microns:

        * Original voxel size is ``(0.290, 0.065, 0.065)``.
        * Scaling factor is ``(1, 4, 4)`` in each dimension.
        * After rescaling the voxel size is ``(0.29 0.26 0.26)``.
    """
    fname = POOCH.fetch(
        "cells3d.tif",
        downloader=pooch.HTTPDownloader(progressbar=True),
    )
    return tifffile.imread(fname)


def mri():
    """MRI dataset of a human head.

    The returned data is a 3D grayscale array with dimensions provided in
    ``(saggital, transverse, coronal)`` order.
    The voxels are isotropic. Voxel size: 0.8 x 0.8 x 0.8 millimeters^3.
    File size: 98.3 MB

    Returns
    -------
    mri: (240, 320, 320) float32 ndarray
        The volumetric images from an MRI of a human head.

    Notes
    -----
    This MRI dataset of a human head was provided by Robert Haase,
    and acquired at University Hospital Carl Gustav Carus
    of the University of Technology, TU Dresden
    as part of academic training of students in the Department of Radiology.

    This dataset, along with several others, has been made available on zenodo:
    https://zenodo.org/record/5090508#.YYNYi5txWtm
    """
    fname = POOCH.fetch(
        "Haase_MRT_tfl3d1.tif",
        downloader=pooch.HTTPDownloader(progressbar=True),
    )
    return tifffile.imread(fname)


def plantseed():
    """3D fluorescence microscopy image of plant seed membranes.

    Arabidopsis ovule primordium.
    Voxel size: 0.1677602 x 0.1677602 x 0.1677602 microns^3
    File size: 28.2 MB

    Returns
    -------
    plantseed: (428, 256, 256) uint8 ndarray
        The volumetric images of an immature plant seed.

    Notes
    -----
    Originally published at https://datadryad.org/stash/dataset/doi:10.5061/dryad.02v6wwq2c
    Converted to tiff and made available here https://zenodo.org/record/5090508#.YYIhd5txWtm
    CC0 Public Domain by Baroux, Célia, University of Zurich, Mendocilla-Sato, Ethel,
    University of Zurich, Autran, Daphné, IRD Montpellier.
    """
    fname = POOCH.fetch(
        "EM_C_6_c0.tif",
        downloader=pooch.HTTPDownloader(progressbar=True),
    )
    return tifffile.imread(fname)
