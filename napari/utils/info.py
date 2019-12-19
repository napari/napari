import platform
import sys

import dask
import numpy
import scipy
import skimage
import vispy
from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

import napari


def sys_info(as_html=False):
    if API_NAME == 'PySide2':
        API_VERSION = PYSIDE_VERSION
    elif API_NAME == 'PyQt5':
        API_VERSION = PYQT_VERSION
    else:
        API_VERSION = ''
    sys_version = sys.version.replace('\n', ' ').strip()

    versions = (
        f"<b>napari</b>: {napari.__version__}<br>"
        f"<b>Platform</b>: {platform.platform()}<br>"
        f"<b>Python</b>: {sys_version}<br>"
        f"<b>{API_NAME}</b>: {API_VERSION}<br>"
        f"<b>Qt</b>: {QtCore.__version__}<br>"
        f"<b>VisPy</b>: {vispy.__version__}<br>"
        f"<b>NumPy</b>: {numpy.__version__}<br>"
        f"<b>SciPy</b>: {scipy.__version__}<br>"
        f"<b>scikit-image</b>: {skimage.__version__}<br>"
        f"<b>Dask</b>: {dask.__version__}<br>"
    )

    sys_info_text = "<br>".join(
        [vispy.sys_info().split("\n")[index] for index in [-4, -3]]
    )

    text = f'{versions}<br>{sys_info_text}'

    if not as_html:
        text = (
            text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
        )
    return text


citation_text = (
    'napari contributors (2019). napari: a '
    'multi-dimensional image viewer for python. '
    'doi:10.5281/zenodo.3555620'
)
