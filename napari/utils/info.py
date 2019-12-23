import platform

import sys

import napari


def sys_info(as_html=False):
    """Gathers relevant module versions for troubleshooting purposes.

    Parameters
    ----------
    as_html : bool
        if True, info will be returned as HTML, suitable for a QTextEdit widget
    """

    sys_version = sys.version.replace('\n', ' ')
    text = (
        f"<b>napari</b>: {napari.__version__}<br>"
        f"<b>Platform</b>: {platform.platform()}<br>"
        f"<b>Python</b>: {sys_version}<br>"
    )

    try:
        from qtpy import API_NAME, PYQT_VERSION, PYSIDE_VERSION, QtCore

        if API_NAME == 'PySide2':
            API_VERSION = PYSIDE_VERSION
        elif API_NAME == 'PyQt5':
            API_VERSION = PYQT_VERSION
        else:
            API_VERSION = ''

        text += (
            f"<b>Qt</b>: {QtCore.__version__}<br>"
            f"<b>{API_NAME}</b>: {API_VERSION}<br>"
        )
    except Exception as e:
        text += f"<b>Qt</b>: Import failed ({e})<br>"

    modules = (
        ('vispy', 'VisPy'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('skimage', 'scikit-image'),
        ('dask', 'Dask'),
    )

    loaded = {}
    for module, name in modules:
        try:
            loaded[module] = __import__(module)
            text += f"<b>{name}</b>: {loaded[module].__version__}<br>"
        except Exception as e:
            text += f"<b>{name}</b>: Import failed ({e})<br>"

    if loaded.get('vispy', False):
        sys_info_text = "<br>".join(
            [
                loaded['vispy'].sys_info().split("\n")[index]
                for index in [-4, -3]
            ]
        ).replace("'", "")
        text += f'<br>{sys_info_text}'

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
